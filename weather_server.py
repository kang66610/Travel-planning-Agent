"""
天气 Agent Web Server — Flask 后端
增强记忆系统：跨对话记忆、RAG 语义检索、自动提取、Markdown 导出
"""
import json
import os
import re
import requests
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
from datetime import datetime

# RAG 引擎（延迟初始化）
import memory_rag

app = Flask(__name__)

# ================= 配置区域 =================
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEATHER_API_KEY = os.environ.get("QWEATHER_API_KEY", "")
QWEATHER_HOST = os.environ.get("QWEATHER_HOST", "n63tehkbee.re.qweatherapi.com")
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
GAODE_API_KEY = os.environ.get("GAODE_API_KEY", "")
MODEL_NAME = "deepseek-v4-flash"
MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_JSON = os.path.join(MEMORY_DIR, "agent_memory.json")
MEMORY_MD = os.path.join(MEMORY_DIR, "agent_memory.md")
CHAT_HISTORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json")
SESSIONS_DIR = os.path.join(MEMORY_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
MAX_ITERATIONS = 5

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ================= 增强记忆系统 =================
# 结构化记忆格式，类似 Claude Code 的 MEMORY.md 设计

def _default_memory() -> dict:
    return {
        "version": 2,
        "user_profile": {
            "name": "",
            "home_city": "",
            "favorite_cities": [],
            "clothing_preference": "normal",
            "disliked_weather": [],
            "activities": [],
            "personality_notes": ""
        },
        "facts": [],
        "conversations": [],
        "query_history": []
    }

def load_memory() -> dict:
    if os.path.exists(MEMORY_JSON):
        try:
            with open(MEMORY_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧版本
            if data.get("version") != 2:
                return _migrate_old_memory(data)
            return data
        except (json.JSONDecodeError, KeyError):
            return _default_memory()
    return _default_memory()

def _migrate_old_memory(old: dict) -> dict:
    """从 v1 格式迁移到 v2"""
    new = _default_memory()
    new["user_profile"]["favorite_cities"] = old.get("favorite_cities", [])
    new["user_profile"]["clothing_preference"] = old.get("clothing_preference", "normal")
    new["query_history"] = old.get("query_history", [])[-50:]
    save_memory(new, rebuild_index=True)
    return new

def save_memory(memory: dict, rebuild_index: bool = False):
    memory["version"] = 2
    with open(MEMORY_JSON, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    _export_memory_md(memory)
    # 仅在明确需要时重建索引（add_fact 已增量更新）
    if rebuild_index and memory_rag.RAG_READY:
        memory_rag.rebuild_index(memory.get("facts", []))

def _next_fact_id(memory: dict) -> int:
    if not memory["facts"]:
        return 1
    return max(f["id"] for f in memory["facts"]) + 1

def add_fact(memory: dict, content: str, category: str = "general",
             importance: str = "medium", source: str = "conversation") -> dict:
    """添加一条事实到记忆库"""
    # 去重：检查是否已有相同内容
    for fact in memory["facts"]:
        if fact["content"].strip() == content.strip():
            fact["last_accessed"] = datetime.now().isoformat()
            fact["access_count"] = fact.get("access_count", 0) + 1
            save_memory(memory)
            return fact

    fact = {
        "id": _next_fact_id(memory),
        "content": content.strip(),
        "category": category,
        "importance": importance,
        "source": source,
        "created": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "access_count": 1
    }
    memory["facts"].append(fact)
    # 增量写入向量索引
    if memory_rag.RAG_READY:
        memory_rag.add_to_index(fact)
    save_memory(memory)
    return fact

def search_memory(memory: dict, query: str) -> list:
    """RAG 混合检索：向量语义 + 关键词匹配"""
    facts = memory.get("facts", [])
    if not facts:
        return []

    if memory_rag.RAG_READY:
        # RAG 模式：语义 + 关键词混合检索
        results = memory_rag.hybrid_search(query, facts, top_k=10)
    else:
        # 降级：纯关键词匹配
        results = []
        query_lower = query.lower()
        keywords = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', query_lower))
        for fact in facts:
            content_lower = fact["content"].lower()
            score = 0
            for kw in keywords:
                if kw in content_lower:
                    score += 2
            if query_lower in fact.get("category", ""):
                score += 1
            if score > 0:
                results.append({**fact, "_score": score})
        results.sort(key=lambda x: -x["_score"])

    # 更新访问计数（不写磁盘，下次 save_memory 时统一保存）
    for r in results[:5]:
        for fact in memory["facts"]:
            if fact["id"] == r["id"]:
                fact["last_accessed"] = datetime.now().isoformat()
                fact["access_count"] = fact.get("access_count", 0) + 1
                break

    return results[:10]

def get_relevant_memories(memory: dict, user_query: str) -> str:
    """获取与当前查询相关的记忆摘要"""
    parts = []

    # 用户画像
    profile = memory.get("user_profile", {})
    profile_items = []
    if profile.get("name"):
        profile_items.append(f"姓名: {profile['name']}")
    if profile.get("home_city"):
        profile_items.append(f"常驻城市: {profile['home_city']}")
    if profile.get("favorite_cities"):
        profile_items.append(f"常查城市: {', '.join(profile['favorite_cities'][:8])}")
    if profile.get("clothing_preference") and profile["clothing_preference"] != "normal":
        pref_map = {"cold_sensitive": "怕冷", "heat_sensitive": "怕热"}
        profile_items.append(f"体质: {pref_map.get(profile['clothing_preference'], profile['clothing_preference'])}")
    if profile.get("disliked_weather"):
        profile_items.append(f"不喜欢的天气: {', '.join(profile['disliked_weather'])}")
    if profile.get("activities"):
        profile_items.append(f"常见活动: {', '.join(profile['activities'][:5])}")
    if profile.get("personality_notes"):
        profile_items.append(f"备注: {profile['personality_notes']}")
    if profile_items:
        parts.append("## 用户画像\n" + "\n".join(f"- {item}" for item in profile_items))

    # 搜索相关事实（跳过空查询）
    if user_query == "__profile_only__":
        relevant = []
    else:
        relevant = search_memory(memory, user_query)
    if relevant:
        fact_lines = []
        for f in relevant[:8]:
            cat_emoji = {"personal": "👤", "preference": "⭐", "plan": "📋",
                         "travel": "✈️", "health": "💊", "work": "💼"}.get(f["category"], "📝")
            fact_lines.append(f"{cat_emoji} {f['content']}")
        parts.append("## 相关记忆\n" + "\n".join(fact_lines))

    # 最近对话摘要
    recent_convs = memory.get("conversations", [])[-3:]
    if recent_convs:
        conv_lines = []
        for c in recent_convs:
            time_str = c.get("time", "")[:16]
            conv_lines.append(f"- [{time_str}] {c.get('summary', '')}")
        parts.append("## 最近对话\n" + "\n".join(conv_lines))

    return "\n\n".join(parts) if parts else "暂无相关记忆"

def auto_extract_facts(memory: dict, user_query: str, assistant_reply: str):
    """用 LLM 自动从对话中提取重要事实"""
    try:
        extract_prompt = f"""分析以下对话，提取关于用户的**重要事实和个人信息**。
只提取确实提到的信息，不要猜测。每条事实用一行中文描述。

用户说: {user_query}
助手回复: {assistant_reply[:500]}

提取规则:
- 用户的姓名、住处、工作等个人信息
- 用户的偏好（怕冷/怕热、喜欢/不喜欢的天气）
- 用户的出行计划、旅行安排
- 用户的日常活动习惯
- 与天气相关的重要上下文

输出格式: 每行一条事实，没有事实则输出"无"。不要编号，不要标点符号。
最多提取 5 条。"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": extract_prompt}],
            max_tokens=300,
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        if text == "无" or not text:
            return

        # 自动分类
        category_map = {
            "住": "personal", "家": "personal", "工作": "work", "公司": "work",
            "怕冷": "preference", "怕热": "preference", "喜欢": "preference", "不喜欢": "preference",
            "出差": "plan", "旅行": "travel", "旅游": "travel", "计划": "plan",
            "跑步": "health", "运动": "health",
        }
        importance_keywords = {"出差", "旅行", "住", "家", "怕冷", "怕热", "过敏", "病"}

        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.、-• ")
            if not line or len(line) < 4:
                continue

            # 分类
            category = "general"
            for kw, cat in category_map.items():
                if kw in line:
                    category = cat
                    break

            # 重要性
            importance = "medium"
            for kw in importance_keywords:
                if kw in line:
                    importance = "high"
                    break

            add_fact(memory, line, category=category, importance=importance, source="auto_extract")

        # 更新用户画像
        _update_profile(memory, user_query, assistant_reply)

    except Exception as e:
        print(f"[Memory] 自动提取失败: {e}")

def _update_profile(memory: dict, user_query: str, assistant_reply: str):
    """根据对话更新用户画像"""
    profile = memory.get("user_profile", {})

    # 提取城市
    cities = re.findall(r'[一-龥]{2,6}', user_query)
    for city in cities:
        if len(city) >= 2 and city not in profile.get("favorite_cities", []):
            # 排除常见非城市词
            skip_words = {"今天", "明天", "昨天", "天气", "怎么样", "如何", "查询", "穿衣",
                          "建议", "预报", "空气", "质量", "指数", "日出", "日落", "预警"}
            if city not in skip_words:
                favs = profile.setdefault("favorite_cities", [])
                if city not in favs and len(favs) < 20:
                    favs.append(city)

    # 提取不喜欢的天气
    if "不喜欢" in user_query or "讨厌" in user_query:
        for w in ["雨", "雪", "风", "热", "冷", "雾霾", "沙尘"]:
            if w in user_query:
                profile.setdefault("disliked_weather", [])
                if w not in profile["disliked_weather"]:
                    profile["disliked_weather"].append(w)

    # 提取活动
    activity_keywords = {"跑步": "跑步", "健身": "健身", "爬山": "爬山", "游泳": "游泳",
                         "骑行": "骑行", "露营": "露营", "徒步": "徒步", "钓鱼": "钓鱼",
                         "打球": "打球", "散步": "散步", "瑜伽": "瑜伽"}
    for kw, act in activity_keywords.items():
        if kw in user_query:
            profile.setdefault("activities", [])
            if act not in profile["activities"]:
                profile["activities"].append(act)

    memory["user_profile"] = profile

def _export_memory_md(memory: dict):
    """导出为人类可读的 Markdown 文件"""
    lines = ["# 🧠 天机 · 记忆库\n"]
    lines.append(f"> 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 用户画像
    profile = memory.get("user_profile", {})
    lines.append("## 👤 用户画像\n")
    if profile.get("name"):
        lines.append(f"- **姓名**: {profile['name']}")
    if profile.get("home_city"):
        lines.append(f"- **常驻城市**: {profile['home_city']}")
    if profile.get("favorite_cities"):
        lines.append(f"- **常查城市**: {', '.join(profile['favorite_cities'][:10])}")
    if profile.get("clothing_preference") and profile["clothing_preference"] != "normal":
        pref_map = {"cold_sensitive": "怕冷", "heat_sensitive": "怕热"}
        lines.append(f"- **体质偏好**: {pref_map.get(profile['clothing_preference'], '')}")
    if profile.get("disliked_weather"):
        lines.append(f"- **不喜欢的天气**: {', '.join(profile['disliked_weather'])}")
    if profile.get("activities"):
        lines.append(f"- **常见活动**: {', '.join(profile['activities'])}")
    if profile.get("personality_notes"):
        lines.append(f"- **备注**: {profile['personality_notes']}")
    lines.append("")

    # 事实记忆
    facts = memory.get("facts", [])
    if facts:
        lines.append("## 📝 记忆事实\n")
        # 按类别分组
        categories = {}
        for f in facts:
            cat = f.get("category", "general")
            categories.setdefault(cat, []).append(f)

        cat_names = {"personal": "👤 个人信息", "preference": "⭐ 偏好",
                     "plan": "📋 计划", "travel": "✈️ 旅行",
                     "health": "💊 健康", "work": "💼 工作", "general": "📝 其他"}

        for cat, cat_facts in categories.items():
            lines.append(f"### {cat_names.get(cat, cat)}\n")
            for f in sorted(cat_facts, key=lambda x: x.get("created", ""), reverse=True):
                imp = "🔴" if f.get("importance") == "high" else "🟡" if f.get("importance") == "medium" else "⚪"
                time_str = f.get("created", "")[:10]
                lines.append(f"- {imp} {f['content']} `[{time_str}]`")
            lines.append("")

    # 最近对话
    convs = memory.get("conversations", [])
    if convs:
        lines.append("## 💬 最近对话摘要\n")
        for c in convs[-10:]:
            time_str = c.get("time", "")[:16]
            lines.append(f"- **[{time_str}]** {c.get('summary', '无摘要')}")
        lines.append("")

    # 统计
    lines.append("## 📊 统计\n")
    lines.append(f"- 事实总数: {len(facts)}")
    lines.append(f"- 对话总数: {len(convs)}")
    lines.append(f"- 查询总数: {len(memory.get('query_history', []))}")
    lines.append(f"- 常查城市: {len(profile.get('favorite_cities', []))} 个")

    with open(MEMORY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def update_query_history(memory: dict, user_query: str, cities: list):
    """更新查询历史"""
    memory.setdefault("query_history", []).append({
        "time": datetime.now().isoformat(),
        "query": user_query,
        "cities": cities
    })
    memory["query_history"] = memory["query_history"][-100:]

def record_conversation(memory: dict, user_query: str, assistant_reply: str, tool_calls: list):
    """记录一次完整对话"""
    # 用 LLM 生成对话摘要
    try:
        summary_prompt = f"""用一句话（20字以内）概括这次对话的核心内容。

用户: {user_query[:200]}
回复: {assistant_reply[:300]}

输出格式: 直接输出摘要句子，不要前缀。"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=60,
            temperature=0.1
        )
        summary = response.choices[0].message.content.strip()
    except Exception:
        summary = user_query[:30]

    # 收集涉及的工具
    tools_used = [tc.get("name", "") for tc in (tool_calls or [])]

    conv = {
        "time": datetime.now().isoformat(),
        "user_query": user_query[:200],
        "summary": summary,
        "tools_used": tools_used,
        "cities_mentioned": [tc.get("result", {}).get("city", "") for tc in (tool_calls or []) if tc.get("result", {}).get("city")]
    }
    memory.setdefault("conversations", []).append(conv)
    # 保留最近 200 条
    memory["conversations"] = memory["conversations"][-200:]
    save_memory(memory)

# ================= 工具缓存 =================
import time as _time

_tool_cache = {}  # key → {"data": ..., "expire": timestamp}
CACHE_TTL = 600   # 缓存有效期 10 分钟

def _cache_get(key: str):
    """获取缓存，过期返回 None"""
    item = _tool_cache.get(key)
    if item and _time.time() < item["expire"]:
        return item["data"]
    _tool_cache.pop(key, None)
    return None

def _cache_set(key: str, data):
    """写入缓存"""
    _tool_cache[key] = {"data": data, "expire": _time.time() + CACHE_TTL}

def _cache_key(func_name: str, *args) -> str:
    """生成缓存 key"""
    return f"{func_name}:{':'.join(str(a) for a in args)}"

# ================= 工具层 =================
def _get_location_id(city: str) -> dict:
    cached = _cache_get(f"geo:{city}")
    if cached:
        return cached
    host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
    geo_url = f"https://{host}/geo/v2/city/lookup?location={city}&key={QWEATHER_API_KEY}"
    geo_res = requests.get(geo_url, timeout=10).json()
    if geo_res.get("code") != "200":
        return None
    locations = geo_res.get("location", [])
    if not locations:
        return None
    loc = locations[0]
    result = {"id": loc["id"], "name": loc["name"], "country": loc["country"]}
    _cache_set(f"geo:{city}", result)
    return result

def get_weather(city: str) -> str:
    try:
        cached = _cache_get(f"get_weather:{city}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/weather/now?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"天气API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        now = res["now"]
        data = {
            "country": loc["country"], "city": loc["name"],
            "weather": now["text"], "temperature": f"{now['temp']}°C",
            "feels_like": f"{now['feelsLike']}°C", "wind_direction": now["windDir"],
            "wind_scale": f"{now['windScale']}级", "humidity": f"{now['humidity']}%"
        }
        result = json.dumps(data, ensure_ascii=False)
        _cache_set(f"get_weather:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def get_forecast(city: str, days: int = 3) -> str:
    try:
        cached = _cache_get(f"get_forecast:{city}:{days}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/weather/3d?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预报API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        forecast_list = []
        for day in res["daily"][:days]:
            forecast_list.append({
                "date": day["fxDate"], "weather_day": day["textDay"],
                "weather_night": day["textNight"], "temp_max": f"{day['tempMax']}°C",
                "temp_min": f"{day['tempMin']}°C", "wind_dir": day["windDirDay"],
                "wind_scale": f"{day['windScaleDay']}级", "humidity": f"{day['humidity']}%"
            })
        result = json.dumps({"city": loc["name"], "forecast": forecast_list}, ensure_ascii=False)
        _cache_set(f"get_forecast:{city}:{days}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def get_air_quality(city: str) -> str:
    """使用 aqicn.org 免费 API 获取空气质量"""
    try:
        cached = _cache_get(f"get_air_quality:{city}")
        if cached:
            return cached
        url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
        res = requests.get(url, timeout=10).json()
        if res.get("status") != "ok":
            return json.dumps({"error": f"空气质量查询失败: {res.get('data', '未知错误')}"}, ensure_ascii=False)
        d = res["data"]
        iaqi = d.get("iaqi", {})
        data = {
            "city": d.get("city", {}).get("name", city),
            "aqi": d.get("aqi"),
            "category": d.get("level", ""),  # aqicn 没有直接的 level 字段
            "pm2_5": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "co": iaqi.get("co", {}).get("v")
        }
        # 根据 AQI 值推算等级
        aqi = data["aqi"]
        if aqi is not None:
            if aqi <= 50: data["category"] = "优"
            elif aqi <= 100: data["category"] = "良"
            elif aqi <= 150: data["category"] = "轻度污染"
            elif aqi <= 200: data["category"] = "中度污染"
            elif aqi <= 300: data["category"] = "重度污染"
            else: data["category"] = "严重污染"
        result = json.dumps(data, ensure_ascii=False)
        _cache_set(f"get_air_quality:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def suggest_clothing(temperature: str, weather: str) -> str:
    try:
        temp_num = int(temperature.replace("°C", "").replace("°", "").strip())
    except (ValueError, AttributeError):
        temp_num = 20
    clothes = []
    if temp_num <= 0:
        clothes = ["厚羽绒服", "保暖内衣", "围巾", "手套", "帽子", "厚靴"]
        advice = "天气严寒，务必做好全身保暖！"
    elif temp_num <= 10:
        clothes = ["厚外套/毛呢大衣", "毛衣", "保暖裤", "围巾"]
        advice = "天气较冷，注意保暖。"
    elif temp_num <= 20:
        clothes = ["薄外套/卫衣", "长袖衬衫", "牛仔裤"]
        advice = "天气微凉，建议穿外套。"
    elif temp_num <= 28:
        clothes = ["短袖/薄T恤", "短裤/薄裙"]
        advice = "天气舒适偏热，轻装出行即可。"
    else:
        clothes = ["短袖", "短裤", "凉鞋", "遮阳帽"]
        advice = "天气炎热，注意防暑防晒！"
    if "雨" in weather:
        clothes.append("雨伞/雨衣")
        advice += "记得带伞！"
    elif "雪" in weather:
        clothes.append("防水靴")
        advice += "注意路面湿滑！"
    elif "风" in weather:
        clothes.append("防风外套")
        advice += "注意防风！"
    data = {"temperature": temperature, "weather": weather, "recommended_clothes": clothes, "advice": advice}
    return json.dumps(data, ensure_ascii=False)

def get_life_index(city: str) -> str:
    try:
        cached = _cache_get(f"get_life_index:{city}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/indices/1d?type=1,2,3,4,5,9&location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"生活指数API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        indices = {}
        type_map = {"1": "运动", "2": "洗车", "3": "紫外线", "4": "旅游", "5": "过敏", "9": "感冒"}
        for item in res.get("daily", []):
            idx_type = type_map.get(item.get("type"), item.get("type"))
            indices[idx_type] = {"level": item.get("level"), "category": item.get("category"), "text": item.get("text")}
        result = json.dumps({"city": loc["name"], "indices": indices}, ensure_ascii=False)
        _cache_set(f"get_life_index:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def get_sun_rise_set(city: str) -> str:
    try:
        cached = _cache_get(f"get_sun_rise_set:{city}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/astronomy/sun?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"天文API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        today = res.get("weatherDaily", {})
        data = {"city": loc["name"], "sunrise": today.get("sunrise", ""), "sunset": today.get("sunset", "")}
        result = json.dumps(data, ensure_ascii=False)
        _cache_set(f"get_sun_rise_set:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def compare_cities(cities: str) -> str:
    try:
        city_list = [c.strip() for c in cities.split(",") if c.strip()][:5]
        if not city_list:
            return json.dumps({"error": "请提供至少一个城市名称"}, ensure_ascii=False)
        results = []
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        for city in city_list:
            loc = _get_location_id(city)
            if not loc:
                results.append({"city": city, "error": "未找到该城市"})
                continue
            url = f"https://{host}/v7/weather/now?location={loc['id']}&key={QWEATHER_API_KEY}"
            res = requests.get(url, timeout=10).json()
            if res.get("code") != "200":
                results.append({"city": city, "error": "API异常"})
                continue
            now = res["now"]
            results.append({
                "city": loc["name"], "temperature": f"{now['temp']}°C",
                "feels_like": f"{now['feelsLike']}°C", "weather": now["text"],
                "humidity": f"{now['humidity']}%", "wind_dir": now["windDir"],
                "wind_scale": f"{now['windScale']}级"
            })
        return json.dumps({"comparison": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def get_weather_warning(city: str) -> str:
    try:
        cached = _cache_get(f"get_weather_warning:{city}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/warning/now?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预警API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        warnings = []
        for item in res.get("warning", []):
            warnings.append({"title": item.get("title", ""), "level": item.get("level", ""), "text": item.get("text", "")})
        result = json.dumps({"city": loc["name"], "has_warning": len(warnings) > 0, "warnings": warnings}, ensure_ascii=False)
        _cache_set(f"get_weather_warning:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

def get_holiday_weather(city: str) -> str:
    try:
        cached = _cache_get(f"get_holiday_weather:{city}")
        if cached:
            return cached
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/weather/7d?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url, timeout=10).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预报API异常，状态码: {res.get('code')}"}, ensure_ascii=False)
        forecast = []
        for day in res.get("daily", []):
            forecast.append({
                "date": day["fxDate"], "weather_day": day["textDay"],
                "weather_night": day["textNight"], "temp_max": f"{day['tempMax']}°C",
                "temp_min": f"{day['tempMin']}°C", "humidity": f"{day['humidity']}%"
            })
        weather_count = {}
        for f in forecast:
            weather_count[f["weather_day"]] = weather_count.get(f["weather_day"], 0) + 1
        trend = max(weather_count, key=weather_count.get) if weather_count else "未知"
        result = json.dumps({"city": loc["name"], "forecast": forecast, "weather_trend": f"未来{len(forecast)}天以{trend}为主"}, ensure_ascii=False)
        _cache_set(f"get_holiday_weather:{city}", result)
        return result
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)

# ================= 旅行工具（高德地图 API） =================
def search_pois(city: str, keyword: str, types: str = "") -> str:
    """搜索城市中的景点、餐厅、酒店等 POI 信息"""
    if not GAODE_API_KEY:
        return json.dumps({"error": "高德地图 API Key 未配置"}, ensure_ascii=False)
    try:
        url = f"https://restapi.amap.com/v3/place/text"
        params = {
            "key": GAODE_API_KEY,
            "keywords": keyword,
            "city": city,
            "output": "json",
            "offset": 10,
            "extensions": "all"
        }
        if types:
            params["types"] = types
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("status") != "1":
            return json.dumps({"error": f"搜索失败: {res.get('info', '未知错误')}"}, ensure_ascii=False)
        pois = []
        for p in res.get("pois", [])[:10]:
            pois.append({
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "address": p.get("address", ""),
                "tel": p.get("tel", ""),
                "location": p.get("location", ""),
                "rating": p.get("biz_ext", {}).get("rating", ""),
                "cost": p.get("biz_ext", {}).get("cost", ""),
                "photos": len(p.get("photos", []))
            })
        return json.dumps({"city": city, "keyword": keyword, "count": len(pois), "pois": pois}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"搜索异常: {str(e)}"}, ensure_ascii=False)

def get_travel_route(origin_city: str, dest_city: str, mode: str = "drive") -> str:
    """获取两个城市之间的出行路线"""
    if not GAODE_API_KEY:
        return json.dumps({"error": "高德地图 API Key 未配置"}, ensure_ascii=False)
    try:
        url = "https://restapi.amap.com/v3/direction/transit/integrated"
        if mode == "drive":
            url = "https://restapi.amap.com/v3/direction/driving"
        elif mode == "walk":
            url = "https://restapi.amap.com/v3/direction/walking"
        elif mode == "bike":
            url = "https://restapi.amap.com/v3/direction/bicycling"

        params = {"key": GAODE_API_KEY, "origin": origin_city, "destination": dest_city, "output": "json"}
        if mode == "transit":
            params["city"] = origin_city

        res = requests.get(url, params=params, timeout=10).json()
        if res.get("status") != "1":
            return json.dumps({"error": f"路线规划失败: {res.get('info', '未知错误')}"}, ensure_ascii=False)

        route = res.get("route", {})
        if mode == "drive":
            paths = route.get("paths", [])
            if paths:
                p = paths[0]
                return json.dumps({
                    "origin": origin_city, "destination": dest_city,
                    "distance": f"{int(p.get('distance', 0)) / 1000:.1f}km",
                    "duration": f"{int(p.get('duration', 0)) / 60:.0f}分钟",
                    "tolls": f"{p.get('tolls', 0)}元",
                    "strategy": p.get("strategy", "")
                }, ensure_ascii=False)
        elif mode == "transit":
            paths = route.get("transits", [])
            if paths:
                p = paths[0]
                return json.dumps({
                    "origin": origin_city, "destination": dest_city,
                    "distance": f"{int(p.get('distance', 0)) / 1000:.1f}km",
                    "duration": f"{int(p.get('duration', 0)) / 60:.0f}分钟",
                    "cost": f"{p.get('cost', 0)}元",
                    "segments": len(p.get("segments", []))
                }, ensure_ascii=False)
        return json.dumps({"error": "未找到路线"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"路线规划异常: {str(e)}"}, ensure_ascii=False)

def search_food(city: str, keyword: str = "美食") -> str:
    """搜索城市中的美食推荐"""
    return search_pois(city, keyword, "050000")

def search_scenic(city: str) -> str:
    """搜索城市中的热门景点"""
    return search_pois(city, "景点", "110000")

def get_geocode(address: str) -> str:
    """地理编码：将地址转换为经纬度坐标"""
    if not GAODE_API_KEY:
        return json.dumps({"error": "高德地图 API Key 未配置"}, ensure_ascii=False)
    try:
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"key": GAODE_API_KEY, "address": address, "output": "json"}
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("status") != "1" or not res.get("geocodes"):
            return json.dumps({"error": "地理编码失败"}, ensure_ascii=False)
        geo = res["geocodes"][0]
        return json.dumps({
            "address": address,
            "location": geo.get("location", ""),
            "formatted_address": geo.get("formatted_address", ""),
            "level": geo.get("level", "")
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"编码异常: {str(e)}"}, ensure_ascii=False)
def save_memory_fact(content: str, category: str = "general", importance: str = "medium") -> str:
    """手动保存一条重要事实到记忆库"""
    global session_memory
    fact = add_fact(session_memory, content, category=category, importance=importance, source="manual")
    return json.dumps({"status": "saved", "fact_id": fact["id"], "content": fact["content"]}, ensure_ascii=False)

def search_memory_facts(query: str) -> str:
    """搜索记忆库中的相关信息"""
    global session_memory
    results = search_memory(session_memory, query)
    if not results:
        return json.dumps({"results": [], "message": "未找到相关记忆"}, ensure_ascii=False)
    items = []
    for r in results:
        items.append({
            "id": r["id"], "content": r["content"],
            "category": r.get("category", ""),
            "importance": r.get("importance", ""),
            "created": r.get("created", "")[:10]
        })
    return json.dumps({"results": items, "count": len(items)}, ensure_ascii=False)

def get_user_profile() -> str:
    """获取用户画像信息"""
    global session_memory
    profile = session_memory.get("user_profile", {})
    return json.dumps(profile, ensure_ascii=False)

def delete_memory_fact(fact_id: int) -> str:
    """删除一条记忆事实"""
    global session_memory
    session_memory["facts"] = [f for f in session_memory["facts"] if f["id"] != fact_id]
    save_memory(session_memory, rebuild_index=True)
    return json.dumps({"status": "deleted", "fact_id": fact_id}, ensure_ascii=False)

# ================= 新增工具 =================
def plan_multi_day_clothing(city: str, days: int = 3) -> str:
    """多日穿衣规划：获取未来多天预报，为每天生成穿衣建议"""
    try:
        # 获取预报
        forecast_json = get_forecast(city, days=days)
        forecast_data = json.loads(forecast_json)
        if "error" in forecast_data:
            return forecast_json

        plan = []
        for day in forecast_data.get("forecast", []):
            temp_max = int(day["temp_max"].replace("°C", ""))
            temp_min = int(day["temp_min"].replace("°C", ""))
            avg_temp = (temp_max + temp_min) // 2
            weather_day = day["weather_day"]

            # 复用穿衣逻辑
            clothes = []
            if avg_temp <= 0:
                clothes = ["厚羽绒服", "保暖内衣", "围巾", "手套"]
            elif avg_temp <= 10:
                clothes = ["厚外套", "毛衣", "保暖裤"]
            elif avg_temp <= 20:
                clothes = ["薄外套/卫衣", "长袖衬衫"]
            elif avg_temp <= 28:
                clothes = ["短袖/薄T恤", "短裤"]
            else:
                clothes = ["短袖", "短裤", "凉鞋"]

            if "雨" in weather_day:
                clothes.append("雨伞")
            if "风" in day.get("wind_dir", ""):
                clothes.append("防风外套")

            plan.append({
                "date": day["date"],
                "weather": weather_day,
                "temp_range": f"{day['temp_min']} ~ {day['temp_max']}",
                "clothes": clothes
            })

        return json.dumps({"city": forecast_data["city"], "days": len(plan), "plan": plan}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"规划失败: {str(e)}"}, ensure_ascii=False)

def check_weather_warnings(cities: str) -> str:
    """批量检查多个城市的天气预警"""
    try:
        city_list = [c.strip() for c in cities.split(",") if c.strip()][:5]
        if not city_list:
            return json.dumps({"error": "请提供城市名称"}, ensure_ascii=False)
        results = []
        for city in city_list:
            warning_json = get_weather_warning(city)
            warning_data = json.loads(warning_json)
            if "error" in warning_data:
                results.append({"city": city, "has_warning": False, "error": warning_data["error"]})
            else:
                results.append(warning_data)
        all_warnings = any(r.get("has_warning") for r in results)
        return json.dumps({"checked_cities": city_list, "has_any_warning": all_warnings, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"检查失败: {str(e)}"}, ensure_ascii=False)

# ================= 工具注册表 =================
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "获取指定城市的当前实时天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_forecast", "description": "获取指定城市未来1-3天的天气预报", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 3}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_air_quality", "description": "获取指定城市的实时空气质量数据", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "suggest_clothing", "description": "根据温度和天气给出穿衣建议", "parameters": {"type": "object", "properties": {"temperature": {"type": "string"}, "weather": {"type": "string"}}, "required": ["temperature", "weather"]}}},
    {"type": "function", "function": {"name": "get_life_index", "description": "获取生活指数（运动、紫外线、洗车、旅游、过敏、感冒）", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_sun_rise_set", "description": "获取日出日落时间", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "compare_cities", "description": "对比多个城市的天气", "parameters": {"type": "object", "properties": {"cities": {"type": "string", "description": "逗号分隔的城市名"}}, "required": ["cities"]}}},
    {"type": "function", "function": {"name": "get_weather_warning", "description": "获取天气预警信息", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_holiday_weather", "description": "获取未来7天天气趋势", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    # === 记忆管理工具 ===
    {"type": "function", "function": {"name": "save_memory_fact", "description": "将重要信息保存到记忆库。当用户提到个人信息、偏好、计划时主动调用。", "parameters": {"type": "object", "properties": {
        "content": {"type": "string", "description": "要记忆的事实内容"},
        "category": {"type": "string", "enum": ["personal", "preference", "plan", "travel", "health", "work", "general"], "description": "类别"},
        "importance": {"type": "string", "enum": ["high", "medium", "low"], "description": "重要程度"}
    }, "required": ["content"]}}},
    {"type": "function", "function": {"name": "search_memory_facts", "description": "搜索记忆库中的相关信息", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_user_profile", "description": "获取用户的画像信息（常驻城市、偏好、活动等）", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "delete_memory_fact", "description": "删除一条记忆事实", "parameters": {"type": "object", "properties": {"fact_id": {"type": "integer", "description": "事实ID"}}, "required": ["fact_id"]}}},
    # === 新增工具 ===
    {"type": "function", "function": {"name": "plan_multi_day_clothing", "description": "为用户规划未来多天的穿衣搭配，每天给出温度范围和穿衣建议", "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名称"},
        "days": {"type": "integer", "description": "规划天数，默认3天", "default": 3}
    }, "required": ["city"]}}},
    {"type": "function", "function": {"name": "check_weather_warnings", "description": "批量检查多个城市是否有天气预警（台风、暴雨、高温等）", "parameters": {"type": "object", "properties": {
        "cities": {"type": "string", "description": "逗号分隔的城市名，如：佛山,广州,深圳"}
    }, "required": ["cities"]}}},
    # === 旅行工具 ===
    {"type": "function", "function": {"name": "search_pois", "description": "搜索城市中的景点、餐厅、酒店等兴趣点", "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名称"},
        "keyword": {"type": "string", "description": "搜索关键词，如：景点、美食、酒店、博物馆"},
        "types": {"type": "string", "description": "POI类型编码，可选：110000景点 050000美食 100000酒店 060000购物 080000交通"}
    }, "required": ["city", "keyword"]}}},
    {"type": "function", "function": {"name": "search_scenic", "description": "搜索城市中的热门景点", "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名称"}
    }, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_food", "description": "搜索城市中的美食推荐", "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名称"},
        "keyword": {"type": "string", "description": "搜索关键词，默认'美食'，可指定如'火锅'、'小吃'"}
    }, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_travel_route", "description": "获取两个城市之间的出行路线（驾车/公交/步行）", "parameters": {"type": "object", "properties": {
        "origin_city": {"type": "string", "description": "出发城市"},
        "dest_city": {"type": "string", "description": "目的地城市"},
        "mode": {"type": "string", "enum": ["drive", "transit", "walk", "bike"], "description": "出行方式", "default": "drive"}
    }, "required": ["origin_city", "dest_city"]}}},
    {"type": "function", "function": {"name": "get_geocode", "description": "地理编码：将地址/地名转换为经纬度坐标", "parameters": {"type": "object", "properties": {
        "address": {"type": "string", "description": "地址或地名，如：北京市天安门"}
    }, "required": ["address"]}}}
]

TOOL_MAP = {
    "get_weather": get_weather, "get_forecast": get_forecast, "get_air_quality": get_air_quality,
    "suggest_clothing": suggest_clothing, "get_life_index": get_life_index,
    "get_sun_rise_set": get_sun_rise_set, "compare_cities": compare_cities,
    "get_weather_warning": get_weather_warning, "get_holiday_weather": get_holiday_weather,
    "save_memory_fact": save_memory_fact, "search_memory_facts": search_memory_facts,
    "get_user_profile": get_user_profile, "delete_memory_fact": delete_memory_fact,
    "plan_multi_day_clothing": plan_multi_day_clothing, "check_weather_warnings": check_weather_warnings,
    "search_pois": search_pois, "search_scenic": search_scenic,
    "search_food": search_food, "get_travel_route": get_travel_route,
    "get_geocode": get_geocode
}

TOOL_EMOJIS = {
    "get_weather": "🌤️", "get_forecast": "📅", "get_air_quality": "🌬️",
    "suggest_clothing": "👔", "get_life_index": "🏃", "get_sun_rise_set": "🌅",
    "compare_cities": "📊", "get_weather_warning": "⚠️", "get_holiday_weather": "🎉",
    "save_memory_fact": "💾", "search_memory_facts": "🔍",
    "get_user_profile": "👤", "delete_memory_fact": "🗑️",
    "plan_multi_day_clothing": "👗", "check_weather_warnings": "🚨",
    "search_pois": "📍", "search_scenic": "🏔️", "search_food": "🍜",
    "get_travel_route": "🚗", "get_geocode": "📌"
}

# ================= 会话管理系统 =================
def _gen_session_id() -> str:
    """生成会话 ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()

def _session_file(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def create_session() -> dict:
    """创建新会话并写入文件"""
    sid = _gen_session_id()
    session = {
        "id": sid,
        "title": "新对话",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "messages": []
    }
    with open(_session_file(sid), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return session

def load_session(session_id: str) -> dict:
    """加载会话"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_session(session: dict):
    """保存会话"""
    session["updated"] = datetime.now().isoformat()
    with open(_session_file(session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

def list_sessions() -> list:
    """列出所有会话（按更新时间倒序），清理空会话"""
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            path = os.path.join(SESSIONS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            msg_count = len(s.get("messages", []))
            if msg_count == 0:
                os.remove(path)
                continue
            sessions.append({
                "id": s["id"],
                "title": s.get("title", "新对话"),
                "created": s.get("created", ""),
                "updated": s.get("updated", ""),
                "msg_count": msg_count
            })
        except Exception:
            continue
    sessions.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return sessions

def delete_session(session_id: str) -> bool:
    """删除会话"""
    path = _session_file(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def rename_session(session_id: str, title: str) -> bool:
    """重命名会话"""
    session = load_session(session_id)
    if not session:
        return False
    session["title"] = title
    save_session(session)
    return True

def auto_title_session(session: dict, first_query: str):
    """用 LLM 自动生成会话标题，直接修改 session 对象"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"用8个字以内概括这次对话主题，只输出标题，不要前缀和标点：\n用户说：{first_query[:100]}"}],
            max_tokens=30,
            temperature=0.1
        )
        title = response.choices[0].message.content.strip()
    except Exception:
        title = first_query[:20]
    session["title"] = title
    print(f"[Session] 自动命名: {title}")
SYSTEM_PROMPT = """你是一个专业的**个人出行规划助手**，具备实时天气感知、景点美食搜索、路线规划、长期记忆等能力。

## 核心定位
你的核心价值：**根据实时天气 + 目的地信息，为用户量身定制出行方案**。不只是查天气，而是成为一个懂天气、懂旅行、懂用户的智能出行管家。

## 核心原则
1. **先想再查**：分析用户意图 → 决定查什么 → 调用工具 → 综合回答
2. **多步推理**：复杂问题分步解决，每次调用工具后判断信息是否充足
3. **天气驱动决策**：所有出行建议必须基于实时天气数据，不要凭空推荐
4. **贴心个性化**：结合用户记忆（偏好、体质、习惯）给出定制化建议

## 工具能力矩阵

### 🌤️ 天气感知层
- get_weather → 实时天气（温度、风力、湿度）
- get_forecast → 未来3天预报
- get_air_quality → 空气质量（AQI、PM2.5）
- check_weather_warnings → 多城市预警检查
- plan_multi_day_clothing → 多日穿衣规划
- get_life_index → 生活指数（运动、紫外线、洗车等）
- get_sun_rise_set → 日出日落时间

### 📍 目的信息层（高德地图 API）
- search_scenic → 搜索景点（评分、门票、照片数）
- search_food → 搜索美食（评分、人均消费）
- search_pois → 通用 POI 搜索（酒店、商场、医院等）
- get_travel_route → 城市间路线规划（驾车/公交/步行）

### 🧠 记忆层
- save_memory_fact → 记住用户信息
- search_memory_facts → 搜索历史记忆
- get_user_profile → 获取用户画像

## 出行规划工作流（核心！）

当用户提到旅行、出行、游玩、攻略、规划等关键词时，执行以下流程：

### 第一步：收集信息（并行调用）
1. **天气数据**：get_weather + get_forecast + check_weather_warnings + get_air_quality
2. **目的地信息**：search_scenic + search_food
3. **出行方式**：如果用户提到出发地 → get_travel_route

### 第二步：智能分析
- 根据天气决定每天适合的活动类型
  - 晴天 25°C+ → 户外景点、骑行、徒步
  - 阴天 15-25°C → 最佳游览日，全天户外
  - 雨天 → 室内景点（博物馆、茶馆、美食街）
  - 高温 35°C+ → 避开正午，早晚出行
  - 有预警 → 调整行程避开危险区域
- 根据景点评分和距离优化路线
- 根据用户体质（记忆）调整强度

### 第三步：生成完整攻略

#### 📅 每日行程表
| 时间 | 活动 | 天气适配 | 交通 |
|------|------|---------|------|
| 上午 | 景点A | ☀️ 适合户外 | 地铁30分钟 |
| 午餐 | 美食B | - | 步行5分钟 |
| 下午 | 景点C | 🌦️ 备选室内 | 打车20分钟 |
| 晚上 | 夜市D | 🌙 夜间出行 | 地铁 |

#### 👔 穿搭与装备
- 每日穿搭（结合温度+天气+活动强度）
- 必带物品清单

#### 🚗 交通方案
- 城市间交通（高铁/飞机/自驾）
- 市内交通（地铁/公交/打车）
- 景点间接驳

#### 💰 预算估算
- 交通费 + 门票 + 餐饮 + 住宿（参考搜索到的价格）

#### ⚠️ 安全提醒
- 天气预警注意事项
- 高温/雨天/大风防护建议

## 回答风格
- 热情专业，像一个资深旅行达人在给朋友推荐
- 数据驱动：所有建议基于实时数据，不凭空编造
- 实用优先：给出具体的时间、地点、价格，不要空泛建议
- 适当 emoji 让攻略生动易读
- 如果记忆中有用户相关信息，主动提及（如"你之前说怕冷..."）"""

def build_system_prompt(memory: dict) -> str:
    parts = [SYSTEM_PROMPT]
    # 只加载用户画像和最近对话摘要，不搜索（空查询无意义）
    memory_context = get_relevant_memories(memory, "__profile_only__")
    if memory_context and memory_context != "暂无相关记忆":
        parts.append(f"\n{memory_context}")
    return "\n".join(parts)

# ================= Flask 路由 =================
session_memory = load_memory()

# 初始化 RAG 向量索引
memory_rag.init_rag()
if memory_rag.RAG_READY:
    if not memory_rag.load_index():
        # 索引不存在，从已有 facts 构建
        memory_rag.rebuild_index(session_memory.get("facts", []))
        print(f"[RAG] 初始索引构建完成: {len(session_memory.get('facts', []))} 条记忆")

session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]

# 启动时清理空会话文件
for fname in os.listdir(SESSIONS_DIR):
    if not fname.endswith(".json"):
        continue
    try:
        with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
            s = json.load(f)
        if len(s.get("messages", [])) == 0:
            os.remove(os.path.join(SESSIONS_DIR, fname))
    except Exception:
        pass

current_session = create_session()  # 当前会话

@app.route("/")
def index():
    return render_template("index.html", gaode_key=GAODE_API_KEY)

@app.route("/api/chat", methods=["POST"])
def chat():
    """处理用户消息，SSE 流式返回 Agent 回复"""
    data = request.json
    user_query = data.get("message", "").strip()
    session_id = data.get("session_id", "")
    if not user_query:
        return jsonify({"error": "消息不能为空"}), 400

    global session_memory, session_messages, current_session

    # 加载或创建会话
    if session_id and session_id != current_session["id"]:
        loaded = load_session(session_id)
        if loaded:
            current_session = loaded
            # 重建 session_messages
            session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]
            for m in current_session.get("messages", []):
                if m.get("role") in ("user", "assistant"):
                    session_messages.append({"role": m["role"], "content": m["content"]})

    # 首条消息自动命名
    is_first = len(current_session.get("messages", [])) == 0

    cities_found = re.findall(r'[一-龥]{2,6}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', user_query)
    update_query_history(session_memory, user_query, cities_found)

    memory_context = get_relevant_memories(session_memory, user_query)
    system_content = SYSTEM_PROMPT
    if memory_context and memory_context != "暂无相关记忆":
        system_content += f"\n\n{memory_context}"
    session_messages[0]["content"] = system_content
    session_messages.append({"role": "user", "content": user_query})
    # 滑动窗口：保留 system prompt + 最近 100 条消息
    if len(session_messages) > 101:
        session_messages = [session_messages[0]] + session_messages[-100:]

    # 保存用户消息到会话
    current_session.setdefault("messages", []).append({
        "role": "user", "content": user_query, "time": datetime.now().isoformat()
    })

    tool_calls_log = []

    def generate():
        nonlocal tool_calls_log
        global current_session
        try:
            for iteration in range(MAX_ITERATIONS):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=session_messages,
                    tools=TOOLS
                )
                response_message = response.choices[0].message
                session_messages.append(response_message)

                if not response_message.tool_calls:
                    final_reply = response_message.content

                    # 首条消息自动命名（在 done 之前完成）
                    if is_first:
                        auto_title_session(current_session, user_query)

                    for char in final_reply:
                        yield f"data: {json.dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'tool_calls': tool_calls_log, 'iterations': iteration + 1, 'session_id': current_session['id'], 'title': current_session.get('title', '新对话')}, ensure_ascii=False)}\n\n"

                    # 保存助手回复到会话
                    current_session["messages"].append({
                        "role": "assistant", "content": final_reply,
                        "tool_calls": tool_calls_log,
                        "time": datetime.now().isoformat()
                    })

                    save_session(current_session)

                    try:
                        record_conversation(session_memory, user_query, final_reply, tool_calls_log)
                        auto_extract_facts(session_memory, user_query, final_reply)
                    except Exception as e:
                        print(f"[Memory] 后处理失败: {e}")
                    return

                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    func = TOOL_MAP.get(func_name)
                    if not func:
                        function_response = json.dumps({"error": f"未知工具: {func_name}"}, ensure_ascii=False)
                    else:
                        function_response = func(**args)

                    emoji = TOOL_EMOJIS.get(func_name, "🔧")
                    try:
                        result_data = json.loads(function_response)
                    except Exception:
                        result_data = {"raw": function_response}

                    tc_info = {"name": func_name, "emoji": emoji, "args": args, "result": result_data}
                    tool_calls_log.append(tc_info)
                    yield f"data: {json.dumps({'type': 'tool', 'name': func_name, 'emoji': emoji}, ensure_ascii=False)}\n\n"

                    session_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": function_response
                    })

            # 超过最大迭代
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=session_messages + [{"role": "user", "content": "请根据已有信息给出最终回答。"}]
            )
            final_content = final_response.choices[0].message.content
            if is_first:
                auto_title_session(current_session, user_query)
            for char in final_content:
                yield f"data: {json.dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'tool_calls': tool_calls_log, 'iterations': MAX_ITERATIONS, 'session_id': current_session['id'], 'title': current_session.get('title', '新对话')}, ensure_ascii=False)}\n\n"
            session_messages.append({"role": "assistant", "content": final_content})
            current_session["messages"].append({
                "role": "assistant", "content": final_content,
                "tool_calls": tool_calls_log, "time": datetime.now().isoformat()
            })
            save_session(current_session)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ================= 会话管理 API =================
@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    return jsonify({"sessions": list_sessions()})

@app.route("/api/sessions/new", methods=["POST"])
def api_new_session():
    global current_session, session_messages
    current_session = create_session()
    session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]
    return jsonify({"session_id": current_session["id"], "title": current_session["title"]})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_load_session(session_id):
    global current_session, session_messages
    session = load_session(session_id)
    if not session:
        return jsonify({"error": "会话不存在"}), 404
    current_session = session
    # 重建 session_messages
    session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]
    for m in session.get("messages", []):
        if m.get("role") in ("user", "assistant"):
            session_messages.append({"role": m["role"], "content": m["content"]})
    return jsonify({"session": session})

@app.route("/api/clear", methods=["POST"])
def clear_chat():
    """清空当前会话的对话历史"""
    global session_messages
    session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]
    return jsonify({"status": "ok"})

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    global current_session, session_messages
    delete_session(session_id)
    if current_session["id"] == session_id:
        current_session = create_session()
        session_messages = [{"role": "system", "content": build_system_prompt(session_memory)}]
    return jsonify({"status": "ok"})

@app.route("/api/sessions/<session_id>/rename", methods=["POST"])
def api_rename_session(session_id):
    data = request.json
    title = data.get("title", "新对话")
    rename_session(session_id, title)
    return jsonify({"status": "ok"})

@app.route("/api/memory", methods=["GET"])
def get_memory():
    """获取完整记忆数据"""
    data = dict(session_memory)
    data["_rag_status"] = {
        "ready": memory_rag.RAG_READY,
        "model": memory_rag.MODEL_NAME if memory_rag.RAG_READY else None,
        "index_size": memory_rag._index.ntotal if memory_rag._index else 0
    }
    return jsonify(data)

@app.route("/api/memory/facts", methods=["GET"])
def get_facts():
    """获取所有记忆事实"""
    return jsonify({"facts": session_memory.get("facts", [])})

@app.route("/api/memory/search", methods=["POST"])
def api_search_memory():
    """搜索记忆"""
    data = request.json
    query = data.get("query", "")
    results = search_memory(session_memory, query)
    return jsonify({"results": results, "count": len(results)})

@app.route("/api/memory/profile", methods=["GET"])
def api_profile():
    """获取用户画像"""
    return jsonify(session_memory.get("user_profile", {}))

@app.route("/api/memory/export", methods=["GET"])
def export_md():
    """导出记忆为 Markdown"""
    _export_memory_md(session_memory)
    return jsonify({"status": "ok", "file": MEMORY_MD})

@app.route("/api/weather-direct", methods=["POST"])
def weather_direct():
    """直接调用天气工具"""
    data = request.json
    city = data.get("city", "北京")
    result = get_weather(city)
    return jsonify(json.loads(result))

if __name__ == "__main__":
    import threading, webbrowser, time

    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
