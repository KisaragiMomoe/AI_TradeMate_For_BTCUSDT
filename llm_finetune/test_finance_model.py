import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "finance_merged_model" # 同目录下的完整模型

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()

test_news = [
    "美联储主席鲍威尔暗示年内不会降息，流动性将持续收紧。", 
    "某黑客组织利用零日漏洞，盗取了某跨链桥价值 1 亿美元的资产。", 
    "MicroStrategy 宣布再次增持 12000 枚比特币，均价 6 万美元。", 
    "美国 SEC 正式批准了贝莱德比特币现货 ETF 的期权交易。", 
    "比特币今日在 6 万至 6.1 万美元区间内窄幅震荡，成交量萎缩。"  
]

def predict_sentiment(news):
    messages = [
        {"role": "system", "content": "你是一个顶级的加密货币金融情绪分类器。请只输出：极度利多、利多、中性、利空、极度利空。"},
        {"role": "user", "content": f"新闻：{news}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

for i, news in enumerate(test_news):
    result = predict_sentiment(news)
    print(f"测试 [{i+1}]: {news}\n🧠 判断: 【 {result} 】\n")