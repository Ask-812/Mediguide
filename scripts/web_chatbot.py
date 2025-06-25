"""
Streamlit Web App for Mediguide Medical Chatbot
Run with: streamlit run scripts/web_chatbot.py
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

st.set_page_config(page_title="Mediguide Medical Chatbot", layout="centered")
st.title("🩺 Mediguide: AI Medical Chatbot")
st.write("Enter your medical question below. This tool provides guideline-based, professional answers with disclaimers. Not a substitute for professional medical advice.")

@st.cache_resource
def load_model():
    model_dir = "models/gpt2-peft-lora"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area("Your question:", height=80)

# Improved preprocessing for keywords vs. statements
preprocessed_input = user_input.strip()
if preprocessed_input:
    words = preprocessed_input.lower().split()
    # If input is a single word or two (likely a keyword)
    if len(words) <= 2 and all(w.isalpha() for w in words):
        preprocessed_input = f"What is {preprocessed_input}?"
    # If input starts with 'i have', 'i feel', 'i am', etc.
    elif re.match(r"i (have|feel|am|suffer from|experience)", preprocessed_input.lower()):
        condition = re.sub(r"^i (have|feel|am|suffer from|experience) ", "", preprocessed_input, flags=re.I)
        preprocessed_input = f"What should I do if I have {condition}?"
    # If input is a question, leave as is
    elif preprocessed_input.endswith("?"):
        pass
    # Otherwise, treat as a general medical question
    else:
        preprocessed_input = f"{preprocessed_input}?"

if st.button("Get Answer") and user_input.strip():
    with st.spinner("Generating answer..."):
        prompt = (
            f"Patient: {preprocessed_input}\n"
            "Doctor (respond concisely, guideline-based, professional, no repetition):"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,  # shorter output
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract only the doctor's answer, remove repeated prompts
            answer_match = re.search(r"Doctor:(.*?)(?:\\nPatient:|\\nDisclaimer:|$)", decoded, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # fallback: remove prompt and disclaimer
                answer = decoded.split('Doctor:')[-1].split('Disclaimer:')[0].strip()
            # Remove repeated sentences
            seen = set()
            filtered = []
            for sent in re.split(r'(?<=[.!?]) +', answer):
                s = sent.strip()
                if s and s not in seen:
                    filtered.append(s)
                    seen.add(s)
            answer = ' '.join(filtered)
        disclaimer = "Disclaimer: This response is for informational purposes only and does not replace professional medical advice."
        st.markdown(f"**Answer:** {answer}")
        st.info(disclaimer)
