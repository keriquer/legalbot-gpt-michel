import streamlit as st
import json
import os

from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# (Optional) Load Hugging Face token from Streamlit secrets
# You can also just use public models without a token for inference-only access
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Load legal decisions from JSON
with open("urteile.json", "r", encoding="utf-8") as f:
    urteile = json.load(f)

# Create prompt template
template = """You are a helpful legal assistant. Based on these court decisions:

{kontext}

Answer this legal question:

{frage}

Only use the facts given in the court decisions above.
"""

prompt = PromptTemplate(input_variables=["kontext", "frage"], template=template)

# Load model from Hugging Face Hub (free)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # very lightweight, works well
    model_kwargs={"temperature": 0.0, "max_length": 512}
)

# Build LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# UI
st.set_page_config(page_title="LegalBot (HF)", page_icon="⚖️")
st.title("⚖️ LegalBot – Free Version via Hugging Face")

frage = st.text_area("📝 Describe your legal case")

if st.button("🔍 Get Prediction"):
    with st.spinner("Analyzing legal cases..."):
        # Naive keyword match (same as before)
        matching = [u for u in urteile if any(word.lower() in u["inhalt"].lower() for word in frage.split())]
        kontext = "\n\n".join([f"- {u['gericht']} ({u['datum']}): {u['inhalt']}" for u in matching[:3]])

        if not kontext:
            st.warning("No relevant court decisions found.")
        else:
            antwort = chain.run(kontext=kontext, frage=frage)
            st.success("📜 Prediction & Reasoning:")
            st.write(antwort)
