import json
import os
import requests
from openai import OpenAI
from datetime import datetime

# ================= 配置区域 =================
DEEPSEEK_API_KEY = "sk-b38b28b852b14b788bebae7c3c6d1aea"
QWEATHER_API_KEY = "5b0d12dc291b4fd49e35d9eb7113d947"
QWEATHER_HOST = "n63tehkbee.re.qweatherapi.com"
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.json")
MAX_ITERATIONS = 5

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ================= 长期记忆 =================
def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "favorite_cities": [],
        "clothing_preference": "normal",
        "focus_metrics": ["temperature", "weather"],
        "query_history": []
    }


def save_memory(memory: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def update_memory(memory: dict, user_query: str, cities: list):
    for city in cities:
        if city not in memory["favorite_cities"]:
            memory["favorite_cities"].append(city)
    memory["query_history"].append({
        "time": datetime.now().isoformat(),
        "query": user_query,
        "cities": cities
    })
    memory["query_history"] = memory["query_history"][-50:]
    save_memory(memory)


# ================= 工具层 =================
def _get_location_id(city: str) -> dict:
    host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
    geo_url = f"https://{host}/geo/v2/city/lookup?location={city}&key={QWEATHER_API_KEY}"
    geo_res = requests.get(geo_url).json()
    if geo_res.get("code") != "200":
        return None
    loc = geo_res["location"][0]
    return {"id": loc["id"], "name": loc["name"], "country": loc["country"]}


def get_weather(city: str) -> str:
    """获取指定城市的当前实时天气"""
    print(f"\n[工具] 查询【{city}】实时天气...")
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/weather/now?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"天气API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        now = res["now"]
        data = {
            "country": loc["country"],
            "city": loc["name"],
            "weather": now["text"],
            "temperature": f"{now['temp']}°C",
            "feels_like": f"{now['feelsLike']}°C",
            "wind_direction": now["windDir"],
            "wind_scale": f"{now['windScale']}级",
            "humidity": f"{now['humidity']}%"
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def get_forecast(city: str, days: int = 3) -> str:
    """获取未来N天的天气预报"""
    print(f"\n[工具] 查询【{city}】未来{days}天天气预报...")
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/weather/3d?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预报API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        forecast_list = []
        for day in res["daily"][:days]:
            forecast_list.append({
                "date": day["fxDate"],
                "weather_day": day["textDay"],
                "weather_night": day["textNight"],
                "temp_max": f"{day['tempMax']}°C",
                "temp_min": f"{day['tempMin']}°C",
                "wind_dir": day["windDirDay"],
                "wind_scale": f"{day['windScaleDay']}级",
                "humidity": f"{day['humidity']}%"
            })
        return json.dumps({"city": loc["name"], "forecast": forecast_list}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def get_air_quality(city: str) -> str:
    """获取指定城市的空气质量"""
    print(f"\n[工具] 查询【{city}】空气质量...")
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/air/now?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"空气API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        now = res["now"]
        data = {
            "city": loc["name"],
            "aqi": now.get("aqi"),
            "category": now.get("category"),
            "pm2_5": now.get("pm2_5"),
            "pm10": now.get("pm10"),
            "so2": now.get("so2"),
            "no2": now.get("no2"),
            "o3": now.get("o3"),
            "co": now.get("co")
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def suggest_clothing(temperature: str, weather: str) -> str:
    """根据温度和天气给出穿衣建议"""
    print(f"\n[工具] 生成穿衣建议...")
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

    data = {
        "temperature": temperature,
        "weather": weather,
        "recommended_clothes": clothes,
        "advice": advice
    }
    return json.dumps(data, ensure_ascii=False)


# ================= 工具注册表 =================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前实时天气，包括温度、体感温度、天气状况、风力风向、湿度等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，支持全球城市，如：北京、东京、纽约、巴黎等"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "获取指定城市未来1-3天的天气预报，包括白天/夜间天气、最高/最低温度、湿度、风力等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "days": {
                        "type": "integer",
                        "description": "预报天数，默认3天，可选1-3",
                        "default": 3
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "获取指定城市的实时空气质量数据，包括AQI指数、PM2.5、PM10、各项污染物浓度、空气质量等级。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_clothing",
            "description": "根据温度和天气状况给出穿衣搭配建议。需要先获取天气数据后再调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "string",
                        "description": "当前温度，如 '25°C'"
                    },
                    "weather": {
                        "type": "string",
                        "description": "天气状况，如 '晴'、'多云'、'小雨' 等"
                    }
                },
                "required": ["temperature", "weather"]
            }
        }
    }
]

TOOL_MAP = {
    "get_weather": get_weather,
    "get_forecast": get_forecast,
    "get_air_quality": get_air_quality,
    "suggest_clothing": suggest_clothing
}


# ================= Agent 核心 =================
SYSTEM_PROMPT = """你是一个专业的全球天气助手 Agent。你具备以下能力：

## 核心原则
1. **先思考再行动**：分析用户意图，制定查询计划
2. **多步推理**：复杂问题需要调用多个工具获取信息后综合回答
3. **主动判断**：根据已有信息判断是否充足，不足时主动补充查询
4. **贴心回答**：结合用户场景（出行、穿衣、运动等）给出实用建议

## 工具使用策略
- 简单天气查询 → 调用 get_weather
- 出差/旅行规划 → get_weather + get_forecast + get_air_quality
- 穿衣建议 → get_weather → suggest_clothing（需要先获取温度和天气再调用）
- 运动/户外建议 → get_weather + get_air_quality
- 未来出行安排 → get_forecast

## 记忆使用
你有用户的长期偏好信息，可以用来个性化回答。例如：
- 如果用户常查某城市，可以主动对比
- 如果用户怕冷，在低温时额外提醒
- 关注用户提到的指标偏好

## 回答风格
- 自然、有温度，像朋友对话
- 数据准确，建议实用
- 适当使用 emoji 让回答更生动"""


def build_system_prompt(memory: dict) -> str:
    parts = [SYSTEM_PROMPT]
    if memory.get("favorite_cities"):
        parts.append(f"\n## 用户偏好\n- 常查城市: {', '.join(memory['favorite_cities'][:5])}")
    if memory.get("clothing_preference") and memory["clothing_preference"] != "normal":
        pref_map = {"cold_sensitive": "怕冷", "heat_sensitive": "怕热", "normal": "一般"}
        parts.append(f"- 体质: {pref_map.get(memory['clothing_preference'], '一般')}")
    if memory.get("focus_metrics"):
        parts.append(f"- 关注指标: {', '.join(memory['focus_metrics'])}")
    return "\n".join(parts)


def execute_tool(tool_call) -> str:
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    func = TOOL_MAP.get(func_name)
    if not func:
        return json.dumps({"error": f"未知工具: {func_name}"}, ensure_ascii=False)
    return func(**args)


def agent_loop(user_query: str, messages: list, memory: dict) -> str:
    # 将用户偏好注入 system prompt
    messages[0]["content"] = build_system_prompt(memory)
    messages.append({"role": "user", "content": user_query})

    # Agent 循环：Plan → Execute → Reflect
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Agent 第 {iteration + 1} 轮 ---")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # 没有工具调用 → 模型认为信息充足，输出最终回复
        if not response_message.tool_calls:
            print(f"[Agent] 回复完成，共迭代 {iteration + 1} 轮")
            return response_message.content

        # 执行所有工具调用
        for tool_call in response_message.tool_calls:
            print(f"[Agent] 调用工具: {tool_call.function.name}")
            function_response = execute_tool(tool_call)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": function_response
            })

    # 超过最大迭代次数，强制输出
    print("[Agent] 达到最大迭代次数，强制输出")
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages + [{"role": "user", "content": "请根据已有信息给出最终回答。"}]
    )
    final_content = final_response.choices[0].message.content
    messages.append({"role": "assistant", "content": final_content})
    return final_content


# ================= 主程序 =================
def main():
    memory = load_memory()
    messages = [
        {"role": "system", "content": build_system_prompt(memory)}
    ]

    print("=" * 55)
    print("  全球天气智能 Agent")
    print("  输入 '退出' 结束对话")
    print("=" * 55)

    while True:
        user_query = input("\n🧑 你: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ['退出', 'quit', 'exit']:
            print("\n👋 再见！已保存你的偏好设置。")
            break

        try:
            reply = agent_loop(user_query, messages, memory)
            print(f"\n🤖 Agent: {reply}")

            # 更新长期记忆
            import re
            cities_found = re.findall(
                r'[一-鿿]{2,6}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',
                user_query
            )
            update_memory(memory, user_query, cities_found)

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
