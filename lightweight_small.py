import json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb
import time
import psutil
import torch

# Initialize WandB
wandb.init(project="therapist-chatbot", name="lora-quantization-training")

# Load and preprocess data (same as before)
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

data_pairs = []
for i in range(0, len(corpus)-1, 2):
    if corpus[i]['role'] == 'user' and corpus[i+1]['role'] == 'therapist':
        input_text = corpus[i]['content']
        output_text = corpus[i + 1]['content']
        data_pairs.append({"input": input_text, "output": output_text})

train_data = data_pairs
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids
    # Set <pad> tokens to -100 to ignore during loss computation
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]
    inputs["labels"] = labels
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Add LoRA configuration
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
target_modules = ["q_proj", "v_proj"]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)
model = get_peft_model(model, peft_config)

# Set TrainingArguments (with quantization applied)
training_args = TrainingArguments(
    output_dir="./results",
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_total_limit=1,
    fp16=True,  # Apply quantization with half-precision
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# Measure total training time
overall_start_time = time.time()
train_result = trainer.train()
overall_end_time = time.time()
total_training_time = (overall_end_time - overall_start_time) / 60

wandb.log({"train/total_training_time_min": total_training_time})
print(f"Total Training Time with LoRA and Quantization: {total_training_time:.2f} minutes")

trainer.save_model("./fine_tuned_therapist_chatbot_lora_quant")
wandb.finish()
