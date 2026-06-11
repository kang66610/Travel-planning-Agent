# 🗺️ 小九 · 智能出行规划 Agent

基于 ReAct 架构的 AI 出行规划助手，集成天气感知、景点搜索、美食推荐、路线规划、RAG 长期记忆等能力，通过一次对话即可生成完整的天气驱动型出行方案。

## ✨ 核心特性

- 🤖 **Agent 架构** — ReAct 循环，18 个工具动态调度，链式推理
- 🧠 **RAG 记忆** — text2vec + FAISS 向量检索，跨对话记住用户偏好
- 🌤️ **天气感知** — 实时天气 + 3日预报 + 空气质量 + 预警检查
- 📍 **地图可视化** — 高德 API POI 标注 + 浏览器实时定位
- 🍜 **美食景点** — 高德 POI 搜索，评分/价格/地址全展示
- 🚗 **路线规划** — 驾车/公交/步行多种出行方式
- 🎤 **语音输入** — Web Speech API，说完自动发送
- 📈 **温度趋势图** — Canvas 双线温度曲线
- 💰 **预算估算** — 根据行程天数自动计算费用
- 📥 **攻略导出** — 一键导出 Markdown 格式攻略
- 🌙 **亮暗主题** — 一键切换，星空/阳光氛围动效
- 💾 **多会话管理** — 持久化存储，自动命名，历史回溯

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | Python / Flask |
| AI | DeepSeek API (deepseek-v4-flash) |
| RAG | sentence-transformers / FAISS |
| 天气 | 和风天气 API / AQICN |
| 地图 | 高德地图 API (Web + JSAPI) |
| 前端 | 原生 HTML/CSS/JS / Canvas |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install flask openai requests sentence-transformers faiss-cpu python-dotenv
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. 启动服务

```bash
python weather_server.py
```

浏览器打开 http://localhost:5000

## 📋 API Key 获取

| API | 用途 | 获取地址 |
|-----|------|---------|
| DeepSeek | AI 模型 | https://platform.deepseek.com/ |
| 和风天气 | 天气数据 | https://dev.qweather.com/ |
| AQICN | 空气质量 | https://aqicn.org/data-platform/token/ |
| 高德地图 | 景点/美食/路线 | https://console.amap.com/ |

## 📁 项目结构

```
├── weather_server.py      # Flask 后端 + Agent 核心
├── memory_rag.py          # RAG 向量检索引擎
├── templates/
│   └── index.html         # 前端页面
├── .env.example           # API Key 模板
├── .gitignore
└── README.md
```

## 📝 License

MIT
