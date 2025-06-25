"""
Script: evaluate_model.py
Description: Evaluates the fine-tuned model on the test set using ROUGE and Perplexity (PPL).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import math

# Load test data (use only first 20 samples for fast evaluation)
df = pd.read_csv('data/medquad_test.csv')

# Load model and tokenizer
model_dir = 'models/gpt2-peft-lora'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = []
ppl_scores = []

for _, row in df.iterrows():
    prompt = f"Patient: {row['question']}\nDoctor:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0], skip_special_tokens=True).split('Doctor:')[-1].split('Disclaimer:')[0].strip()
    # ROUGE
    rouge = scorer.score(row['answer'], answer)
    rouge_scores.append(rouge['rougeL'].fmeasure)
    # Perplexity (truncate answer to 512 tokens)
    encodings = tokenizer(row['answer'], return_tensors='pt', truncation=True, max_length=512)
    max_length = model.config.n_positions
    stride = 256
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        input_ids = encodings.input_ids[:, i:i+stride]
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if outputs.loss is not None and math.isfinite(outputs.loss.item()):
                neg_log_likelihood = outputs.loss * input_ids.size(1)
                nlls.append(neg_log_likelihood)
    if nlls and encodings.input_ids.size(1) > 0:
        ppl = torch.exp(torch.stack(nlls).sum() / encodings.input_ids.size(1)).item()
        if math.isfinite(ppl):
            ppl_scores.append(ppl)

if ppl_scores:
    print(f"Avg ROUGE-L: {np.mean(rouge_scores):.4f}")
    print(f"Avg Perplexity: {np.mean(ppl_scores):.2f}")
else:
    print(f"Avg ROUGE-L: {np.mean(rouge_scores):.4f}")
    print("Avg Perplexity: N/A (all values were invalid)")
