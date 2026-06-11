import json
import os
import re
import requests
import threading
import tkinter as tk
from tkinter import scrolledtext
from openai import OpenAI
from datetime import datetime
import sys

# ================= 高 DPI 支持 =================
if sys.platform == 'win32':
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)

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


def get_life_index(city: str) -> str:
    """获取指定城市的生活指数（运动、紫外线、洗车、旅游等）"""
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/indices/1d?type=1,2,3,4,5,9&location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"生活指数API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        indices = {}
        type_map = {
            "1": "运动", "2": "洗车", "3": "紫外线",
            "4": "旅游", "5": "过敏", "9": "感冒"
        }
        for item in res.get("daily", []):
            idx_type = type_map.get(item.get("type"), item.get("type"))
            indices[idx_type] = {
                "level": item.get("level"),
                "category": item.get("category"),
                "text": item.get("text")
            }

        data = {
            "city": loc["name"],
            "indices": indices
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def get_sun_rise_set(city: str) -> str:
    """获取指定城市的日出日落时间"""
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/astronomy/sun?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"天文API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        today = res.get("weatherDaily", {})
        data = {
            "city": loc["name"],
            "sunrise": today.get("sunrise", ""),
            "sunset": today.get("sunset", ""),
            "daylight_duration": today.get("dayLength", ""),
            "sun_up_time": today.get("sunUp", ""),
            "sun_down_time": today.get("sunDown", "")
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def compare_cities(cities: str) -> str:
    """同时对比多个城市的天气（城市用逗号分隔，如：北京,上海,广州）"""
    try:
        city_list = [c.strip() for c in cities.split(",") if c.strip()]
        if not city_list:
            return json.dumps({"error": "请提供至少一个城市名称"}, ensure_ascii=False)
        if len(city_list) > 5:
            city_list = city_list[:5]

        results = []
        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")

        for city in city_list:
            loc = _get_location_id(city)
            if not loc:
                results.append({"city": city, "error": "未找到该城市"})
                continue

            url = f"https://{host}/v7/weather/now?location={loc['id']}&key={QWEATHER_API_KEY}"
            res = requests.get(url).json()
            if res.get("code") != "200":
                results.append({"city": city, "error": "API异常"})
                continue

            now = res["now"]
            results.append({
                "city": loc["name"],
                "temperature": f"{now['temp']}°C",
                "feels_like": f"{now['feelsLike']}°C",
                "weather": now["text"],
                "humidity": f"{now['humidity']}%",
                "wind_dir": now["windDir"],
                "wind_scale": f"{now['windScale']}级"
            })

        data = {"comparison": results}
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def get_weather_warning(city: str) -> str:
    """获取指定城市的天气预警信息"""
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")
        url = f"https://{host}/v7/warning/now?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预警API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        warnings = []
        for item in res.get("warning", []):
            warnings.append({
                "title": item.get("title", ""),
                "level": item.get("level", ""),
                "type": item.get("type", ""),
                "sender": item.get("sender", ""),
                "publish_time": item.get("pubTime", ""),
                "text": item.get("text", "")
            })

        data = {
            "city": loc["name"],
            "has_warning": len(warnings) > 0,
            "warning_count": len(warnings),
            "warnings": warnings
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


def get_holiday_weather(city: str) -> str:
    """获取指定城市节假日天气（当前月份的天气趋势）"""
    try:
        loc = _get_location_id(city)
        if not loc:
            return json.dumps({"error": f"未找到城市【{city}】"}, ensure_ascii=False)

        host = QWEATHER_HOST.replace("https://", "").replace("http://", "").strip(" /[]")

        # 获取未来7天预报作为节假日参考
        url = f"https://{host}/v7/weather/7d?location={loc['id']}&key={QWEATHER_API_KEY}"
        res = requests.get(url).json()
        if res.get("code") != "200":
            return json.dumps({"error": f"预报API异常，状态码: {res.get('code')}"}, ensure_ascii=False)

        forecast = []
        for day in res.get("daily", []):
            forecast.append({
                "date": day["fxDate"],
                "weather_day": day["textDay"],
                "weather_night": day["textNight"],
                "temp_max": f"{day['tempMax']}°C",
                "temp_min": f"{day['tempMin']}°C",
                "wind_dir": day.get("windDirDay", ""),
                "humidity": f"{day['humidity']}%"
            })

        # 统计天气趋势
        weather_count = {}
        for f in forecast:
            w = f["weather_day"]
            weather_count[w] = weather_count.get(w, 0) + 1

        trend = max(weather_count, key=weather_count.get) if weather_count else "未知"

        data = {
            "city": loc["name"],
            "days": len(forecast),
            "forecast": forecast,
            "weather_trend": f"未来{len(forecast)}天以{trend}为主",
            "suitable_outdoor": trend in ["晴", "多云", "阴"]
        }
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)


# ================= 工具注册表 =================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：北京、东京、纽约"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "获取指定城市未来1-3天的天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "days": {"type": "integer", "description": "预报天数，1-3天", "default": 3}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "获取指定城市的实时空气质量数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_clothing",
            "description": "根据温度和天气给出穿衣建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "string", "description": "当前温度，如 '25°C'"},
                    "weather": {"type": "string", "description": "天气状况，如 '晴'、'小雨'"}
                },
                "required": ["temperature", "weather"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_life_index",
            "description": "获取指定城市的生活指数，包括运动、紫外线、洗车、旅游、过敏、感冒等指数",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sun_rise_set",
            "description": "获取指定城市的日出日落时间，适合规划户外活动",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cities",
            "description": "同时对比多个城市的天气情况，城市用逗号分隔",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {"type": "string", "description": "城市列表，逗号分隔，如：北京,上海,广州"}
                },
                "required": ["cities"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_warning",
            "description": "获取指定城市的天气预警信息，如台风、暴雨、高温等预警",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_holiday_weather",
            "description": "获取指定城市未来7天天气趋势，适合节假日出行参考",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

TOOL_MAP = {
    "get_weather": get_weather,
    "get_forecast": get_forecast,
    "get_air_quality": get_air_quality,
    "suggest_clothing": suggest_clothing,
    "get_life_index": get_life_index,
    "get_sun_rise_set": get_sun_rise_set,
    "compare_cities": compare_cities,
    "get_weather_warning": get_weather_warning,
    "get_holiday_weather": get_holiday_weather
}

TOOL_EMOJIS = {
    "get_weather": "🌤️",
    "get_forecast": "📅",
    "get_air_quality": "🌬️",
    "suggest_clothing": "👔",
    "get_life_index": "🏃",
    "get_sun_rise_set": "🌅",
    "compare_cities": "📊",
    "get_weather_warning": "⚠️",
    "get_holiday_weather": "🎉"
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
- 穿衣建议 → get_weather → suggest_clothing
- 运动/户外建议 → get_weather + get_air_quality + get_life_index
- 未来出行安排 → get_forecast
- 日出日落查询 → get_sun_rise_set
- 多城市对比 → compare_cities
- 天气预警 → get_weather_warning
- 节假日出行 → get_holiday_weather

## 回答风格
- 自然、有温度，像朋友对话
- 数据准确，建议实用
- 适当使用 emoji 让回答更生动"""


def build_system_prompt(memory: dict) -> str:
    parts = [SYSTEM_PROMPT]
    if memory.get("favorite_cities"):
        parts.append(f"\n## 用户偏好\n- 常查城市: {', '.join(memory['favorite_cities'][:5])}")
    return "\n".join(parts)


def execute_tool(tool_call) -> tuple:
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    func = TOOL_MAP.get(func_name)
    if not func:
        return json.dumps({"error": f"未知工具: {func_name}"}, ensure_ascii=False), func_name
    result = func(**args)
    return result, func_name


def agent_loop(user_query: str, messages: list, memory: dict) -> tuple:
    messages[0]["content"] = build_system_prompt(memory)
    messages.append({"role": "user", "content": user_query})

    tool_calls_log = []

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            return response_message.content, tool_calls_log

        for tool_call in response_message.tool_calls:
            function_response, tool_name = execute_tool(tool_call)
            emoji = TOOL_EMOJIS.get(tool_name, "🔧")

            try:
                args = json.loads(tool_call.function.arguments)
                result_data = json.loads(function_response)
                tool_calls_log.append({
                    "name": tool_name,
                    "emoji": emoji,
                    "args": args,
                    "result": result_data
                })
            except Exception:
                tool_calls_log.append({
                    "name": tool_name,
                    "emoji": emoji,
                    "args": {"raw": tool_call.function.arguments},
                    "result": {"raw": function_response}
                })

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": function_response
            })

    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages + [{"role": "user", "content": "请根据已有信息给出最终回答。"}]
    )
    final_content = final_response.choices[0].message.content
    messages.append({"role": "assistant", "content": final_content})
    return final_content, tool_calls_log


# ================= 桌面 UI =================
class OvalButton(tk.Canvas):
    """胶囊形（圆角矩形）按钮组件"""
    def __init__(self, parent, text, command=None, bg="#6366f1", fg="white",
                 font=("Microsoft YaHei UI", 11, "bold"), width=100, height=38, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent.cget("bg"),
                         highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg
        self.fg_color = fg
        self.font = font
        self.w = width
        self.h = height
        self.text = text

        self._draw(bg)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _draw(self, bg, fg=None):
        self.delete("all")
        r = self.h // 2  # 圆角半径
        self._round_rect(0, 0, self.w, self.h, r, fill=bg, outline="")
        self.create_text(self.w//2, self.h//2, text=self.text,
                         fill=fg or self.fg_color, font=self.font)

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        """绘制圆角矩形"""
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_click(self, event):
        if self.command:
            self.command()

    def _on_enter(self, event):
        self.configure(cursor="hand2")
        if self.bg_color == "#f1f5f9":
            self._draw("#6366f1", "white")
        else:
            self._draw("#4f46e5")

    def _on_leave(self, event):
        self._draw(self.bg_color)


class WeatherAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能天气 Agent")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)
        self.root.configure(bg="#f5f7fa")

        self.memory = load_memory()
        self.messages = [{"role": "system", "content": build_system_prompt(self.memory)}]
        self.thinking_shown = False

        self.create_widgets()

    def create_widgets(self):
        # 使用 Grid 布局确保输入框始终可见
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)  # 聊天区域可扩展

        # 标题区域
        title_frame = tk.Frame(self.root, bg="#f5f7fa")
        title_frame.grid(row=0, column=0, sticky="ew", padx=25, pady=(15, 10))

        tk.Label(
            title_frame,
            text="智能天气 Agent",
            font=("Microsoft YaHei UI", 26, "bold"),
            bg="#f5f7fa",
            fg="#1e293b"
        ).pack()

        tk.Label(
            title_frame,
            text="您的专属天气助手  ·  全球天气查询  ·  穿衣建议",
            font=("Microsoft YaHei UI", 11),
            bg="#f5f7fa",
            fg="#64748b"
        ).pack(pady=(6, 0))

        # 快捷按钮区域
        quick_frame = tk.Frame(self.root, bg="#ffffff", relief=tk.FLAT, bd=0,
                               highlightbackground="#e2e8f0", highlightthickness=1)
        quick_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=(0, 10))

        # 城市按钮
        cities_frame = tk.Frame(quick_frame, bg="#ffffff")
        cities_frame.pack(fill=tk.X, padx=15, pady=(12, 5))

        cities = [("北京", "北京"), ("上海", "上海"), ("广州", "广州"), ("深圳", "深圳"), ("佛山", "佛山")]
        for i, (text, city) in enumerate(cities):
            btn = OvalButton(
                cities_frame,
                text=text,
                bg="#6366f1",
                fg="white",
                font=("Microsoft YaHei UI", 11, "bold"),
                width=150,
                height=38,
                command=lambda c=city: self.quick_query(f"{c}今天天气怎么样？")
            )
            btn.pack(side=tk.LEFT, padx=4, pady=5, expand=True, fill=tk.X)

        # 功能按钮 - 第一行
        self.features_frame = tk.Frame(quick_frame, bg="#ffffff")
        self.features_frame.pack(fill=tk.X, padx=15, pady=(5, 4))

        # 功能按钮 - 第二行
        self.features_frame2 = tk.Frame(quick_frame, bg="#ffffff")
        self.features_frame2.pack(fill=tk.X, padx=15, pady=(0, 12))

        self.update_feature_buttons()

        # 聊天区域 - 使用 Grid row=2, weight=1 可扩展
        chat_frame = tk.Frame(self.root, bg="#ffffff", relief=tk.FLAT, bd=0,
                              highlightbackground="#e2e8f0", highlightthickness=1)
        chat_frame.grid(row=2, column=0, sticky="nsew", padx=25, pady=(0, 10))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Microsoft YaHei UI", 11),
            bg="#fafbfc",
            fg="#1e293b",
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=12,
            state=tk.DISABLED,
            cursor="arrow",
            insertbackground="#6366f1"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 配置文本标签
        self.chat_display.tag_configure("user_name",
                                        foreground="#4f46e5",
                                        font=("Microsoft YaHei UI", 11, "bold"))
        self.chat_display.tag_configure("bot_name",
                                        foreground="#7c3aed",
                                        font=("Microsoft YaHei UI", 11, "bold"))
        self.chat_display.tag_configure("tool_name",
                                        foreground="#059669",
                                        font=("Microsoft YaHei UI", 10, "bold"))
        self.chat_display.tag_configure("separator",
                                        foreground="#e2e8f0")

        # 显示欢迎信息
        self.show_welcome()

        # 输入区域 - 使用 Grid row=3 固定在底部
        input_frame = tk.Frame(self.root, bg="#f5f7fa")
        input_frame.grid(row=3, column=0, sticky="ew", padx=25, pady=(0, 15))

        input_container = tk.Frame(input_frame, bg="#ffffff", relief=tk.FLAT, bd=0,
                                   highlightbackground="#e2e8f0", highlightthickness=1)
        input_container.pack(fill=tk.X)

        self.user_input = tk.Entry(
            input_container,
            font=("Microsoft YaHei UI", 12),
            bg="#ffffff",
            fg="#1e293b",
            relief=tk.FLAT,
            bd=0,
            insertbackground="#6366f1"
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(16, 12), pady=12)
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.focus_set()

        send_btn = OvalButton(
            input_container,
            text="发送",
            bg="#6366f1",
            fg="white",
            font=("Microsoft YaHei UI", 11, "bold"),
            width=80,
            height=38,
            command=self.send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=(0, 10), pady=10)

        clear_btn = OvalButton(
            input_container,
            text="清空",
            bg="#e2e8f0",
            fg="#64748b",
            font=("Microsoft YaHei UI", 10),
            width=70,
            height=36,
            command=self.clear_chat
        )
        clear_btn.pack(side=tk.RIGHT, padx=(0, 5), pady=11)

    def get_default_city(self):
        """获取默认城市：优先使用最近查询的城市"""
        history = self.memory.get("query_history", [])
        if history:
            last = history[-1]
            cities = last.get("cities", [])
            if cities:
                return cities[0]
        fav = self.memory.get("favorite_cities", [])
        if fav:
            return fav[0]
        return "北京"

    def update_feature_buttons(self):
        """根据最近查询的城市更新功能按钮"""
        city = self.get_default_city()

        # 清空旧按钮
        for w in self.features_frame.winfo_children():
            w.destroy()
        for w in self.features_frame2.winfo_children():
            w.destroy()

        features = [("3天预报", f"{city}未来三天天气预报"),
                    ("空气质量", f"{city}空气质量如何"),
                    ("穿衣建议", f"{city}今天适合穿什么衣服？")]

        for text, query in features:
            btn = OvalButton(
                self.features_frame,
                text=text,
                bg="#f1f5f9",
                fg="#475569",
                font=("Microsoft YaHei UI", 10),
                width=200,
                height=35,
                command=lambda q=query: self.quick_query(q)
            )
            btn.pack(side=tk.LEFT, padx=4, pady=4, expand=True, fill=tk.X)

        features2 = [("生活指数", f"{city}生活运动指数怎么样"),
                     ("日出日落", f"{city}今天日出日落时间"),
                     ("多城市对比", f"{city},上海,广州天气对比"),
                     ("天气预警", f"{city}有天气预警吗"),
                     ("节假日天气", f"{city}未来7天天气趋势")]

        for text, query in features2:
            btn = OvalButton(
                self.features_frame2,
                text=text,
                bg="#f0fdf4",
                fg="#166534",
                font=("Microsoft YaHei UI", 9),
                width=150,
                height=32,
                command=lambda q=query: self.quick_query(q)
            )
            btn.pack(side=tk.LEFT, padx=3, pady=4, expand=True, fill=tk.X)

    def show_welcome(self):
        """显示欢迎信息"""
        welcome = """欢迎使用智能天气 Agent！

我可以帮你查询：
  实时天气 / 3天预报 / 空气质量
  穿衣建议 / 生活指数 / 日出日落
  多城市对比 / 天气预警 / 节假日天气

点击上方按钮或直接输入问题即可开始。"""
        self.append_chat("Agent", welcome, "bot_name")

    def quick_query(self, query):
        """快捷查询"""
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, query)
        self.send_message()

    def send_message(self, event=None):
        """发送消息"""
        message = self.user_input.get().strip()
        if not message:
            return

        self.user_input.delete(0, tk.END)
        self.append_chat("你", message, "user_name")

        self.user_input.configure(state=tk.DISABLED)
        self.thinking_shown = True
        self.append_chat("Agent", "正在思考...", "bot_name")

        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()

    def process_message(self, message):
        """处理消息"""
        try:
            cities_found = re.findall(
                r'[一-鿿]{2,6}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',
                message
            )
            update_memory(self.memory, message, cities_found)

            final_reply, tool_calls = agent_loop(
                message,
                self.messages,
                self.memory
            )

            self.root.after(0, self.remove_thinking)
            self.root.after(50, lambda: self.show_results(tool_calls, final_reply))
            self.root.after(100, self.update_feature_buttons)

        except Exception as e:
            self.root.after(0, self.remove_thinking)
            self.root.after(50, lambda: self.append_chat("错误", str(e), "user_name"))

        finally:
            self.root.after(0, lambda: self.user_input.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.user_input.focus_set())

    def remove_thinking(self):
        """移除思考提示"""
        if not self.thinking_shown:
            return

        self.chat_display.configure(state=tk.NORMAL)
        content = self.chat_display.get("1.0", tk.END)

        lines = content.split("\n")
        if len(lines) > 8:
            # 找到倒数第二个分隔线的位置
            separator_count = 0
            for i in range(len(lines) - 1, -1, -1):
                if "─" * 20 in lines[i]:
                    separator_count += 1
                    if separator_count == 2:
                        # 删除从这个位置到末尾的内容
                        self.chat_display.delete(f"{i + 1}.0", tk.END)
                        break

        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.thinking_shown = False

    def show_results(self, tool_calls, final_reply):
        """显示结果"""
        if tool_calls:
            for tc in tool_calls:
                self.append_chat("工具", f"{tc['emoji']} {tc['name']}", "tool_name")

        self.append_chat("Agent", final_reply, "bot_name")

    def append_chat(self, sender, message, tag):
        """添加消息"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n{sender}:\n", tag)
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.insert(tk.END, "─" * 50 + "\n", "separator")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def clear_chat(self):
        """清空对话"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.messages = [{"role": "system", "content": build_system_prompt(self.memory)}]
        self.show_welcome()


def main():
    root = tk.Tk()
    app = WeatherAgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
