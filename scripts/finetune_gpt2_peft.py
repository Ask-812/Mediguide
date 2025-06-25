"""
Script: finetune_gpt2_peft.py
Description: Fine-tunes a GPT-2 model on the medquad dataset using PEFT (LoRA) and BitsAndBytes for efficient training.
Adheres to clinical guidelines, uses professional language, and includes disclaimers in outputs.
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import bitsandbytes as bnb

# Load data
def load_data(split):
    df = pd.read_csv(f'data/medquad_{split}.csv')
    # Combine question and answer for training
    df['text'] = 'Patient: ' + df['question'] + '\nDoctor: ' + df['answer'] + '\nDisclaimer: This response is for informational purposes only and does not replace professional medical advice.'
    return Dataset.from_pandas(df[['text']])

train_dataset = load_data('train')
val_dataset = load_data('val')

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Use a small subset for quick testing
train_dataset = train_dataset.select(range(100))
val_dataset = val_dataset.select(range(20))

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Load model without 8-bit quantization for compatibility
model = AutoModelForCausalLM.from_pretrained('gpt2')
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/gpt2-peft-lora",
    per_device_train_batch_size=2,  # Reduced batch size for CPU
    per_device_eval_batch_size=2,
    num_train_epochs=0.1,  # Very short run for fast testing
    logging_steps=5,  # More frequent logging
    learning_rate=2e-4,
    fp16=False,  # Disable fp16 for CPU
    report_to="none",
    push_to_hub=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("models/gpt2-peft-lora")
tokenizer.save_pretrained("models/gpt2-peft-lora")
print("Model fine-tuned and saved to models/gpt2-peft-lora/")
