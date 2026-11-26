import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "finance_lora_model_v2"   # 你的微调产出文件夹
save_path = "finance_merged_model"       # 合并后输出的文件夹

print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("合并中...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

print("保存合并后的模型...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 合并成功！保存在: {save_path}")