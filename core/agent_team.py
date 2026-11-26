import json
import re
from openai import OpenAI
from openai import AsyncOpenAI

class ChiefTraderAgent:
    def __init__(self, model_name = "qwen2.5:14b"):
        self.client = AsyncOpenAI(
            api_key = "ollama", 
            base_url = "http://localhost:11434/v1",
        )
        self.model_name = model_name

    async def make_decision(self, tech_pred_return, obi, is_bull_market, fng_value, news_headline, whale_alert = "无异常"):
        """
        首席交易员：汇总所有情报，做出一票否决权
        """
        system_prompt = """你是一个华尔街顶级的加密货币量化交易员。你的任务是作为最后一道安全闸门，审核量化模型的交易信号。
【规则】：
1. 如果技术面预测做多，但新闻极其利空或出现巨鲸抛售警报，你必须否决（返回 0）。
2. 如果技术面预测做空，但市场极度恐慌且出现巨鲸抄底警报，你必须否决（返回 0）。
3. 只有当你认为技术面信号与宏观环境、盘口力量匹配时，才批准。
请严格输出 JSON 格式：{"decision": 1 / -1 / 0, "reason": "简明扼要的决策理由"}"""

        user_prompt = f"""
【技术面模型预测】: 预测收益 {tech_pred_return * 100:.4f}% (正数偏多，负数偏空)
【当前大趋势】: {"牛市" if is_bull_market else "熊市"}
【微观盘口(OBI)】: {obi:.2f}
【宏观恐慌指数(FNG)】: {fng_value}
【最新新闻快讯】: {news_headline}
【链上巨鲸异动】: {whale_alert}
"""
        try:
            response = await self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 0.1,
                max_tokens = 150,
                extra_body = {"keep_alive": "24h"} 
            )
            res_text = response.choices[0].message.content.strip()
            
            # 提取 JSON
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return int(result.get("decision", 0)), result.get("reason", "")
            return 0, "解析失败，默认观望"
        except Exception as e:
            print(f"❌ 首席交易员连接失败: {e}")
            return 0, "大模型断线，安全起见观望"