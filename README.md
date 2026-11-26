# 🤖 TradeMate: 多智能体协同的加密货币量化交易引擎

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

TradeMate 是一个全栈、端到端的加密货币高频量化交易系统。摒弃了传统的“死板指标”，创新性地采用了 **“LightGBM 多因子预测 + QLoRA 微调 1.5B 情绪大模型 + AI 首席交易员一票否决权”** 的多智能体（Multi-Agent）架构，并配备极速 Web 监控大屏。

> ⚠️ **高风险警告**：本项目仅供 AI 工程、大模型微调与量化交易学习交流使用，不构成任何投资建议。实盘有风险，请务必在**币安 Demo 模拟盘**下运行测试！

---

## ✨ 核心特性 (Features)

- 🧠 **大模型定制化微调**：使用 QLoRA 技术对 Qwen2.5-1.5B 进行微调。模型能在 0.2 秒内精准识别宏观新闻情绪（极度利多、利空等）。
- 📈 **LightGBM 量化预测**：融合 18+ 维度的技术面、量价、K线形态因子，输出未来收益率预测。
- ⚡ **高频 WebSocket 引擎**：直连币安底层行情流，毫秒级接收 K线与订单簿（Order Book），杜绝 REST API 轮询延迟。
- 🛡️ **全链路风控体系**：
  - **AI 首席交易员 (Chief Trader)**：结合新闻、盘口 OBI（买卖压力比）、链上巨鲸异动，对交易信号进行终极审核。
  - **动态 ATR 止损**：自适应市场波动率，计算最合理的止损线。
  - **高频保本巡检器**：浮盈达 1% 时，自动将止损线移至开仓价，锁定利润。
  - **全局熔断机制**：当日亏损超 5%，系统强行休眠防上头。
- 📊 **流式监控大屏**：基于 Gradio 构建的机构级监控面板，实时展示 AI 裁决、预测值与盘口数据。

---

## 🏗️ 系统架构

```text
├── core/
│   ├── agent_team.py     # 微调后的 1.5B 金融情绪 Agent
│   ├── exchange.py       # 币安原生 API 与算法单封装
│   ├── feature_engine.py # 18+ 技术指标工程
│   └── backtest.py       # 历史回测引擎
├── strategy/
│   └── ma_cross.py       # 基础策略 (可扩展)
├── week2_finance_lora/   # (可选) 大模型微调脚本与数据集
├── dashboard.py          # Gradio Web 监控大屏
├── live_trader.py        # 异步实盘交易主引擎
├── train_lightgbm.py     # LightGBM 训练脚本
└── requirements.txt      # 依赖清单
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Python 3.10+，创建独立虚拟环境：
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

*(💡 提示：若使用 NVIDIA 显卡，建议先前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 Torch，再执行上述依赖安装)*

### 2. 部署本地大模型 (Ollama)

本项目使用极速的 1.5B 模型作为 AI 盘手。如果你不想自己训练，可以直接下载我们合并好的模型；如果你想复现训练过程，请进入 `llm_finetune/` 文件夹：

**方法 A（直接部署模型）**
1. 安装并启动 [Ollama](https://ollama.com/)。
2. 进入 `llm_finetune/` 文件夹，执行模型注册：
   `ollama create finance-expert -f Modelfile`

**方法 B（从零微调，硬核玩家）**
1. 确保本地已安装 Ollama 并拉取了 `qwen2.5:14b` 和 `qwen2.5:1.5b`。
2. 进入 `llm_finetune/` 文件夹。
3. 运行 `python generate_data_with_14b.py`（调用 14B 模型合成 500 条均衡金融数据）。
4. 运行 `python train_finance_lora.py`（基于 Qwen2.5-1.5B 进行 QLoRA 微调）。
5. 运行 `python merge_finance_model.py` 合并权重。
6. 最后执行 `ollama create finance-expert -f Modelfile` 注册模型。

### 3. 配置环境变量
在根目录新建 `.env` 文件（**切勿泄露此文件**），填入你的币安 [Demo 模拟盘](https://demo.binance.com/) API：
```env
BINANCE_API_KEY=your_demo_api_key
BINANCE_SECRET_KEY=your_demo_secret_key
```

### 4. 启动引擎与大屏
你需要打开两个终端（并激活虚拟环境）。

**终端 1：启动高频异步交易引擎**
```bash
python live_trader.py
```

**终端 2：启动可视化监控大屏**
```bash
python dashboard.py
```
打开浏览器访问 `http://127.0.0.1:7860` 即可实时监控。

---

## 🧪 模型训练与回测 (Optional)

- **训练 LightGBM 模型**：运行 `python train_lightgbm.py` （自动拉取 5000 根 K 线训练并评估）。
- **微调 1.5B 金融模型**：进入 `week2_finance_lora`，运行 `generate_data_with_14b.py` 合成数据，再运行 `train_finance_lora.py`。

---

## 💡 踩坑总结 (DevLog)

在构建这套系统时，解决了一系列真实世界的工程难题：
1. **量化重叠计算**：将预测目标从 `shift(-4)` 改为 `shift(-1)`，彻底消灭了多单重叠收益的虚假回测。
2. **API 限频与断线重连**：使用 WebSocket 长连接，加入心跳检测与 10 秒超时自动重连机制。
3. **交易所订单拒绝 (-2022)**：引入 1.5 秒延迟等待，彻底解决 `ReduceOnly` 平仓单被拒的问题。
4. **孤儿单清理**：后台协程每 5 秒扫描一次仓位，若空仓则强制清扫所有残留委托单。

## 最后，祝你玩的开心！