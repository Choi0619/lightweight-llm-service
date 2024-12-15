import json
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import wandb
import time
import psutil
import torch

# Initialize WandB
wandb.init(project="therapist-chatbot", name="gpt-neo-1.3B-training")

# Load corpus.json data
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Prepare input-output pairs
data_pairs = []
for i in range(0, len(corpus)-1, 2):  # Process as pairs of user and therapist
    if corpus[i]['role'] == 'user' and corpus[i+1]['role'] == 'therapist':
        input_text = corpus[i]['content']  # User input
        output_text = corpus[i + 1]['content']  # Therapist response
        data_pairs.append({"input": input_text, "output": output_text})

# Use all data as the training set
train_data = data_pairs

# Convert dataset to Hugging Face format
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Set pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids
    
    # Set <pad> tokens to -100 to ignore during loss calculation
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]
    
    inputs["labels"] = labels
    return inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)

# Use DataCollatorWithPadding
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_total_limit=1,
    fp16=False,
    report_to="wandb",  # Log to WandB
)

# Initialize custom trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_memory_usage_list = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # Start runtime measurement
        start_time = time.time()

        # Measure memory and GPU usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # Append GPU memory usage to the list
        if torch.cuda.is_available():
            self.gpu_memory_usage_list.append(gpu_memory_usage)
        
        # Compute loss and get outputs
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        # End runtime measurement
        end_time = time.time()
        runtime = end_time - start_time
        
        # Log to WandB
        wandb.log({
            "train/loss": loss.item(),
            "train/runtime": runtime,
            "train/memory_usage_MB": memory_usage,
            "train/gpu_memory_usage_MB": gpu_memory_usage,
        })
        
        return (loss, outputs) if return_outputs else loss

    def log_average_gpu_memory_usage(self):
        # Calculate average GPU memory usage and log to WandB
        if self.gpu_memory_usage_list:
            avg_gpu_memory_usage = sum(self.gpu_memory_usage_list) / len(self.gpu_memory_usage_list)
            wandb.log({"train/average_gpu_memory_usage_MB": avg_gpu_memory_usage})
            print(f"Average GPU Memory Usage: {avg_gpu_memory_usage:.2f} MB")

# Start total training time measurement
overall_start_time = time.time()

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# Start training
train_result = trainer.train()

# Calculate and log total training time
overall_end_time = time.time()
total_training_time = (overall_end_time - overall_start_time) / 60  # Convert to minutes
wandb.log({"train/total_training_time_min": total_training_time})

# Print total training time to console
print(f"Total Training Time: {total_training_time:.2f} minutes")

# Log average GPU memory usage
trainer.log_average_gpu_memory_usage()

# Save the model
trainer.save_model("./fine_tuned_therapist_chatbot")

# Check training log history
df = pd.DataFrame(trainer.state.log_history)
print(df)  # Output log history to check recorded loss values

# End WandB logging
wandb.finish()
