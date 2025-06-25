"""
Streamlit Web App for Mediguide Medical Chatbot
Run with: streamlit run scripts/web_chatbot.py
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# --- UI CONFIG ---
st.set_page_config(page_title="Mediguide Medical Chatbot", layout="wide", page_icon="🩺")

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/medical-doctor.png", width=80)
st.sidebar.title("Mediguide Chatbot")
st.sidebar.markdown("""
**AI-Assisted Medical Q&A**

- Guideline-based, professional answers
- HIPAA-compliant data handling
- Powered by GPT-2 + PEFT (LoRA)

**Try these examples:**
- What are the symptoms of diabetes?
- How is hypertension treated?
- fever
- I have chest pain
""")
st.sidebar.markdown("---")
st.sidebar.info("This tool is for informational purposes only and does not replace professional medical advice.")

# --- MAIN PAGE ---
st.title("🩺 Mediguide: AI Medical Chatbot")
st.markdown("""
Enter your medical question below. This tool provides guideline-based, professional answers with disclaimers. Not a substitute for professional medical advice.
""")

@st.cache_resource
def load_model():
    model_id = "Ask-812/mediguide-gpt2-peft-lora"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

example_questions = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "fever",
    "I have chest pain"
]

def set_example(q):
    st.session_state.user_input = q

col1, col2 = st.columns([3, 1])
with col1:
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''
    user_input = st.text_area("Your question:", height=80, placeholder="Type your medical question here...", value=st.session_state.user_input, key="user_input")
with col2:
    st.markdown("**Example Questions:**")
    for q in example_questions:
        st.button(q, key=q, on_click=set_example, args=(q,))

# Improved preprocessing for keywords vs. statements
preprocessed_input = user_input.strip() if user_input else ""
if preprocessed_input:
    words = preprocessed_input.lower().split()
    if len(words) <= 2 and all(w.isalpha() for w in words):
        preprocessed_input = f"What is {preprocessed_input}?"
    elif re.match(r"i (have|feel|am|suffer from|experience)", preprocessed_input.lower()):
        condition = re.sub(r"^i (have|feel|am|suffer from|experience) ", "", preprocessed_input, flags=re.I)
        preprocessed_input = f"What should I do if I have {condition}?"
    elif preprocessed_input.endswith("?"):
        pass
    else:
        preprocessed_input = f"{preprocessed_input}?"

if st.button("Get Answer") and user_input.strip():
    with st.spinner("Generating answer..."):
        prompt = preprocessed_input  # Only the question, no Patient/Doctor
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=120,  # Increased from 60 to 120 for more complete answers
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    temperature=0.7
                )
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                # Remove any Patient/Doctor/Disclaimer prefixes if present
                answer = re.sub(r"(Patient:|Doctor:|Disclaimer:)", "", decoded, flags=re.I).strip()
                seen = set()
                filtered = []
                for sent in re.split(r'(?<=[.!?]) +', answer):
                    s = sent.strip()
                    # Only keep sentences that end with ., !, or ?
                    if s and s[-1] in '.!?' and s not in seen:
                        filtered.append(s)
                        seen.add(s)
                answer = ' '.join(filtered)
            if not answer or len(answer) < 5:
                st.warning("Sorry, I couldn't generate a reliable answer. Please try rephrasing your question.")
            else:
                st.markdown(f"**Answer:** {answer}")
                disclaimer = "Disclaimer: This response is for informational purposes only and does not replace professional medical advice."
                st.info(disclaimer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Mediguide &copy; 2025 | Created by Arnav</div>", unsafe_allow_html=True)
