# 🤖 TradeMate: 多智能体协同的加密货币量化交易引擎

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-MIT-green)

TradeMate 是一个全栈、端到端的加密货币高频量化交易系统。摒弃了传统的“死板指标”，创新性地采用了 **“LightGBM 多因子预测 + QLoRA 微调 1.5B 情绪大模型 + AI 首席交易员一票否决权”** 的多智能体（Multi-Agent）架构，并配备极速 Web 监控大屏。

> ⚠️ **高风险警告**：本项目仅供 AI 工程、大模型微调与量化交易学习交流使用，不构成任何投资建议。实盘有风险，请务必在**币安 Demo 模拟盘**下运行测试，切勿直接投入实盘资金！

---

## ✨ 核心特性 (Features)

- 🧠 **大模型定制化微调**：在 `llm_finetune` 目录中，使用 QLoRA 技术对 Qwen2.5-1.5B 进行微调。模型能在 0.2 秒内精准识别宏观新闻情绪（极度利多、利空等）。
- 📈 **LightGBM 量化预测**：融合 18+ 维度的技术面、量价、K线形态因子，输出未来收益率预测。
- ⚡ **高频 WebSocket 引擎**：直连币安底层行情流，毫秒级接收 K线与订单簿（Order Book），杜绝 REST API 轮询延迟。
- 🛡️ **全链路风控体系**：
  - **AI 首席交易员**：结合新闻、盘口 OBI（买卖压力比）、链上巨鲸异动，对交易信号进行终极审核。
  - **动态 ATR 止损**：自适应市场波动率，计算最合理的止损线。
  - **高频保本巡检器**：浮盈达 1% 时，自动将止损线移至开仓价，锁定利润。
  - **全局熔断机制**：当日亏损超 5%，系统强行休眠防上头。
- 📊 **流式监控大屏**：基于 Gradio 构建的机构级监控面板，实时展示 AI 裁决、预测值与盘口数据。

---

## 🏗️ 系统架构 (System Architecture)

项目采用了清晰的前后端与训练/推理分离架构：

```text
├── core/                  # 核心算法层
│   ├── agent_team.py      # 微调后的 1.5B 金融情绪 Agent
│   ├── backtest.py        # 严谨的历史回测引擎
│   ├── exchange.py        # 币安底层 API 与原生算法单封装
│   └── feature_engine.py  # 18+ 维度技术指标与因子工程
├── llm_finetune/          # 大模型微调模块 (QLoRA)
│   ├── generate_data_with_14b.py  # 调用 14B 模型生成金融数据集
│   ├── train_finance_lora.py      # QLoRA 微调脚本
│   ├── merge_finance_model.py     # 合并 LoRA 权重脚本
│   ├── test_finance_model.py      # 本地模型评估脚本
│   └── Modelfile                  # Ollama 部署配置文件
├── dashboard.py           # Gradio Web 监控大屏
├── live_trader.py         # 高频异步实盘交易主引擎
├── train_lightgbm.py      # LightGBM 训练脚本
├── lgb_trading_model.txt  # 预训练好的 LightGBM 模型文件
└── requirements.txt       # 依赖清单
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Python 3.10+，创建独立虚拟环境并安装依赖。
先在终端执行 `python -m venv .venv`。
Windows 系统激活命令：`.\.venv\Scripts\activate`。
Mac/Linux 系统激活命令：`source .venv/bin/activate`。
最后执行 `pip install -r requirements.txt` 安装依赖。
*(💡 提示：若使用 NVIDIA 显卡，建议先前往 PyTorch 官网安装对应 CUDA 版本的 Torch，再执行依赖安装。)*

### 2. 配置环境变量
在项目根目录新建一个 `.env` 文件（**切勿泄露此文件**），填入你的币安 Demo 模拟盘 API。
文件内容为：
`BINANCE_API_KEY=your_demo_api_key`
`BINANCE_SECRET_KEY=your_demo_secret_key`

### 3. 部署本地大模型 (Ollama)
本系统依赖极速的 1.5B 情绪模型作为 AI 盘手。
1. 安装并启动 Ollama。
2. 进入 `llm_finetune/` 文件夹。
3. 如果你已经下载了合并后的 `.gguf` 文件，直接在该目录下执行终端命令 `ollama create finance-expert -f Modelfile` 注册模型。
4. 如果你没有现成的模型文件，请参考下方的“模型微调”自行训练。

### 4. 启动引擎与大屏
你需要打开两个终端窗口（确保都处于虚拟环境）。
**终端 1**：输入 `python live_trader.py` 启动高频异步交易引擎。
**终端 2**：输入 `python dashboard.py` 启动可视化监控大屏。
打开浏览器访问 `http://127.0.0.1:7860` 即可实时监控实盘运行状态。

---

## 🧪 模型训练 (Optional)

### 训练 LightGBM 预测模型
如果你修改了特征或想重新训练，请在根目录终端输入：`python train_lightgbm.py`。
运行完毕后会自动覆盖根目录的 `lgb_trading_model.txt`。

### 从零微调 1.5B 金融情绪模型
如需复现大模型微调过程，请在 `llm_finetune/` 目录下依次执行：
1. 确保本地已拉取 `qwen2.5:14b` 和 `qwen2.5:1.5b`。
2. 执行 `python generate_data_with_14b.py` 生成 500 条均衡金融数据。
3. 执行 `python train_finance_lora.py` 执行 QLoRA 微调。
4. 执行 `python merge_finance_model.py` 合并权重。
5. 最后执行 `ollama create finance-expert -f Modelfile` 导入 Ollama。

---

## 💡 踩坑总结 (DevLog)

在开发与实盘调试过程中，本系统解决了以下真实世界的工程难题：
1. **量化重叠计算**：将预测目标从 `shift(-4)` 改为 `shift(-1)`，彻底消灭了多单重叠收益的虚假回测陷阱。
2. **API 限频与假死**：WebSocket 长连接加入心跳检测与 10 秒超时自动重连机制，确保行情不间断。
3. **交易所订单拒绝 (-2022)**：引入 `asyncio.sleep` 等待开仓生效，解决 `ReduceOnly` 止损单被拒的 Bug。
4. **孤儿单清理**：后台巡检协程每 5 秒扫描一次，若空仓则强制撤单，防止委托堆积。
5. **大模型推理延迟**：从 14B 替换为微调后的 1.5B，推理速度提升数十倍，满足高频微观决策需求。
