import streamlit as st
import json
import os

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text2text-generation",
    model="declare-lab/flan-alpaca-base",
    max_length=512,
    temperature=0.0
)

llm = HuggingFacePipeline(pipeline=pipe)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# OPTIONAL: falls du ein Token verwendest (empfohlen bei Rate Limits)
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# 📂 Lade die Gerichtsurteile
with open("urteile.json", "r", encoding="utf-8") as f:
    urteile = json.load(f)

# 🧠 Prompt-Template
template = """You are a helpful legal assistant. Based on these court decisions:

{kontext}

Answer this legal question:

{frage}

Only use the facts given in the court decisions above.
"""

prompt = PromptTemplate(
    input_variables=["kontext", "frage"],
    template=template
)

# 🔍 Wähle ein kostenloses Modell von Hugging Face
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # alternativ: flan-t5-base
    model_kwargs={"temperature": 0.0, "max_length": 512}
)

# 🔗 Chain zusammenbauen
chain = LLMChain(llm=llm, prompt=prompt)

# 🌐 Streamlit UI
st.set_page_config(page_title="LegalBot (HF)", page_icon="⚖️")
st.title("⚖️ LegalBot")

frage = st.text_area("📝 Describe your legal case")

if st.button("🔍 Get Prediction"):
    with st.spinner("Analyzing legal cases..."):
        # 🔎 Einfache Schlüsselwortsuche
        matching = [
            u for u in urteile if any(word.lower() in u["inhalt"].lower() for word in frage.split())
        ]
        kontext = "\n\n".join([
            f"- {u['gericht']} ({u['datum']}): {u['inhalt']}" for u in matching[:3]
        ])

        if not kontext:
            st.warning("⚠️ No relevant court decisions found.")
        else:
            antwort = chain.run({"kontext": kontext, "frage": frage})
            st.success("📜 Prediction & Reasoning:")
            st.write(antwort)
