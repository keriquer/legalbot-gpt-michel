import streamlit as st
import requests
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# --- Fetch recent court cases directly from OpenLegalData.io API ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=5"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

# --- LLM pipeline setup ---
pipe = pipeline(
    "text2text-generation",
    model="declare-lab/flan-alpaca-base",
    max_length=512,
    temperature=0.0
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Streamlit UI ---
st.title("⚖️ LegalBot mit Open Legal Data (deutsch)")

frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch")

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere echte Gerichtsurteile..."):
        # Simple keyword match
        matching = [
            c for c in cases
            if 'text' in c and any(word.lower() in c['text'].lower() for word in frage.split())
        ]
        if not matching:
            st.warning("Keine relevanten Entscheidungen gefunden.")
        else:
            kontext = "\n\n".join(
                [f"- {c['court']} ({c['date_decided']}): {c['text'][:200]}..." for c in matching]
            )
            prompt = f"Du bist ein hilfreicher Rechtsassistent. Basierend auf diesen Gerichtsurteilen:\n\n{kontext}\n\nBeantworte folgende Frage:\n{frage}"
            antwort = llm(prompt)
            st.success("📜 Prognose & Begründung:")
            st.write(antwort)
