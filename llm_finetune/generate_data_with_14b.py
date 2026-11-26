import json
import random
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:14b"

dataset = []
seen_news = set() 

labels = ["极度利多", "利多", "中性", "利空", "极度利空"]
COUNT_PER_LABEL = 100

print("🚀 脚本已启动，准备呼叫 14B 模型...", flush=True)

for current_label in labels:
    print(f"\n--- 正在生成【{current_label}】类别的新闻 ---", flush=True)
    count = 0
    
    while count < COUNT_PER_LABEL:
        try:
            system_prompt = f"""你是一个加密货币金融新闻编辑。
请生成一条【{current_label}】的逼真加密货币新闻。
要求：
1. 字数在 20-50 字之间。
2. 必须非常隐晦或直接地体现出【{current_label}】的特征。
3. 只输出 JSON 格式，例如：{{"news": "比特币突破十万美元大关。"}}
绝对不要输出任何除了 JSON 以外的文字！"""

            temp = random.uniform(0.7, 1.0) 
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个只输出 JSON 格式的 API，不要包含任何 markdown 标记。"},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=temp,
                max_tokens=150 # 👈 加大输出长度，防止截断
            )
            
            res_text = response.choices[0].message.content
            if not res_text:
                print("⚠️ 模型返回了空文本，重试中...")
                continue
                
            res_text = res_text.strip()
            
            # 👈 核心修复：用正则强行抠出 JSON 大括号里的内容
            import re
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if not json_match:
                print(f"⚠️ 模型没有输出 JSON，原始输出为: {res_text}")
                continue
                
            json_str = json_match.group(0)
            data = json.loads(json_str)
            news_text = data["news"]
            
            if news_text in seen_news:
                continue
                
            seen_news.add(news_text)
            
            messages_format = {
                "messages": [
                    {"role": "system", "content": "你是一个顶级的加密货币金融情绪分类器。请只输出：极度利多、利多、中性、利空、极度利空。"},
                    {"role": "user", "content": f"新闻：{news_text}"},
                    {"role": "assistant", "content": current_label}
                ]
            }
            dataset.append(messages_format)
            count += 1
            print(f"✅ [{current_label}] 已生成 {count}/{COUNT_PER_LABEL} 条 | {news_text[:15]}...", flush=True)
            
        except Exception as e:
            print(f"❌ 报错了: {e}")
            import time
            time.sleep(2)

random.shuffle(dataset)

with open("finance_sentiment_14b.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n🎉 数据集生成完毕！已保存为 finance_sentiment_14b.jsonl", flush=True)