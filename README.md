# Mediguide: AI-Assisted Medical Chatbot

[Live Demo](https://mediguidebot.streamlit.app/) | [HuggingFace Model](https://huggingface.co/Ask-812/mediguide-gpt2-peft-lora)

## Project Highlights

- **Conversational AI**: Answers medical questions with guideline-based, professional responses and disclaimers.
- **Modern NLP**: Fine-tunes GPT-2 using PEFT (LoRA) for efficient, resource-friendly training.
- **Evaluation**: Reports ROUGE-L (0.0736) and Perplexity (20.40) on a real medical Q&A dataset.
- **Web Demo**: Interactive Streamlit app for easy, browser-based chatbot access.
- **HIPAA-compliant**: All data anonymized and handled with privacy best practices.

## Overview

Mediguide is a conversational AI system designed to provide preliminary, guideline-adherent responses to medical questions. It demonstrates and compares three fine-tuning strategies for decoder-only transformer models using HuggingFace Transformers, PEFT, and BitsAndBytes.

## Features

- Ingests free-text medical questions and generates professional, guideline-based answers with disclaimers.
- Supports Prompt Tuning, PEFT (LoRA/QLoRA/Adapters), and full fine-tuning.
- Evaluates models on ROUGE, PPL, latency, and model size.
- Generates a PDF report with results and recommendations.
- Ensures HIPAA-equivalent anonymization.

## Project Structure

- `data/` — Datasets (anonymized)
- `models/` — Model checkpoints and configs (not included in repo; see HuggingFace link above)
- `scripts/` — Training, fine-tuning, and inference scripts
- `evaluation/` — Evaluation scripts and metrics
- `reporting/` — PDF report generation

## Setup

1. Create a Python virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   pip install streamlit
   ```

## Usage

- See scripts in `scripts/` for training, fine-tuning, and inference.
- Run evaluation scripts in `evaluation/`.
- Generate the PDF report using scripts in `reporting/`.
- Run the web app:
  ```cmd
  streamlit run scripts/web_chatbot.py
  ```

## Sample Output

**Q:** What is glaucoma?

**A:** Glaucoma is a group of diseases that can damage the eye's optic nerve and result in vision loss and blindness. ...

**Disclaimer:** This response is for informational purposes only and does not replace professional medical advice.

## Deployment

- The chatbot can be deployed as a web app ([Streamlit Live Demo](https://mediguidebot.streamlit.app/)) or as an API (FastAPI) for integration with other systems.
- For production, consider using a GPU-enabled cloud service for faster inference.

## Results Table

| Fine-Tuning Method | ROUGE-L | Perplexity | Model Size | Notes                        |
| ------------------ | ------- | ---------- | ---------- | ---------------------------- |
| LoRA/PEFT (GPT-2)  | 0.0736  | 20.40      | ~500MB     | Efficient, resource-friendly |

## Summary

Mediguide demonstrates a full ML pipeline: data cleaning, efficient fine-tuning, robust evaluation, and a user-friendly web demo. It is ready for portfolio, academic, or professional use.

## Disclaimer

This tool is for informational purposes only and does not replace professional medical advice.
