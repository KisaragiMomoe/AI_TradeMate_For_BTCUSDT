import gradio as gr
import json
import os

def read_state():
    if os.path.exists("live_state.json"):
        try:
            with open("live_state.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

def get_dashboard_data():
    state = read_state()
    
    price = state.get("price", 0.0)
    obi = state.get("obi", 0.5) * 100
    fng = state.get("fng", 50)
    news = state.get("news", "等待新闻...")
    pred = state.get("pred", 0.0)
    ai_dec = state.get("ai_decision", 2)
    ai_reason = state.get("ai_reason", "等待 AI 裁决...")
    
    price_str = f"${price:,.2f}"
    obi_str = f"{obi:.2f}%"
    
    dec_map = {1: "🟢 批准做多", -1: "🔴 批准做空", 0: "🛑 一票否决", 2: "💤 观望中"}
    ai_dec_str = dec_map.get(ai_dec, "未知")
    
    return price_str, obi_str, fng, news, pred, ai_dec_str, ai_reason

# 使用暗黑主题风格
with gr.Blocks(title="TradeMate 量化大屏", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🤖 TradeMate 机构级量化监控大屏")
    gr.Markdown("基于 LightGBM 预测 + 专属 1.5B 金融模型多智能体协同系统")
    
    with gr.Row():
        with gr.Column():
            price_box = gr.Textbox(label="实时 BTC 价格", value="等待中...", interactive=False)
        with gr.Column():
            obi_box = gr.Textbox(label="微观盘口 (OBI)", value="等待中...", interactive=False)
        with gr.Column():
            fng_box = gr.Number(label="恐慌贪婪指数 (FNG)", value=50, interactive=False)
            
    with gr.Row():
        pred_box = gr.Number(label="LightGBM 预测收益率 (%)", value=0.0, interactive=False)
        ai_dec_box = gr.Textbox(label="首席 AI 交易员裁决", value="观望中", interactive=False)
        
    news_box = gr.Textbox(label="最新宏观新闻", value="等待中...", interactive=False)
    ai_reason_box = gr.Textbox(label="AI 裁决理由", value="等待中...", interactive=False, lines=3)
    
    # 用 Timer 自动刷新界面
    timer = gr.Timer(2) # 每 2 秒刷新一次
    timer.tick(fn=get_dashboard_data, inputs=[], outputs=[
        price_box, obi_box, fng_box, news_box, pred_box, ai_dec_box, ai_reason_box
    ])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)