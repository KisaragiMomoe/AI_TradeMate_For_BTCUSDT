import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 基础配置
model_id = "Qwen/Qwen2.5-1.5B-Instruct" # 👈 换成 1.5B 模型，下载极快
output_dir = "./finance_lora_model_v2"

# 2. 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map = "auto")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 3. LoRA 配置
lora_config = LoraConfig(
    r = 8, lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 处理数据 (应用 Chat 模板，并做 Loss Mask)
def format_dataset(example):
    user_text = tokenizer.apply_chat_template([example["messages"][0], example["messages"][1]], tokenize = False, add_generation_prompt = True)
    assistant_text = example["messages"][2]["content"] + tokenizer.eos_token
    
    user_ids = tokenizer(user_text, add_special_tokens = False).input_ids
    assistant_ids = tokenizer(assistant_text, add_special_tokens = False).input_ids
    
    input_ids = user_ids + assistant_ids
    labels = [-100] * len(user_ids) + assistant_ids # 只对回答计算 loss
    
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

print("加载数据...")
dataset = load_dataset("json", data_files="finance_sentiment_14b.jsonl", split = "train")
tokenized_dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 4, # 1.5B 模型显存占用小，可以设大一点
    gradient_accumulation_steps = 2,
    learning_rate = 2e-4,
    num_train_epochs = 3, # 500条数据跑3轮
    logging_steps = 10,
    bf16 = True,
    optim = "paged_adamw_8bit",
    save_strategy = "epoch"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True, label_pad_token_id = -100)
)

print("🚀 开始微调 1.5B 金融情绪模型...")
trainer.train()

print("✅ 微调完成，保存权重...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)