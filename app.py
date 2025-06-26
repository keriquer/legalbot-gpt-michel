import streamlit as st
import requests
from transformers import pipeline

# --- Fetch recent court cases from OpenLegalData.io ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=5"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

st.title("⚖️ LegalBot mit Open Legal Data (deutsch)")
frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'Kündigung')")

@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "google/flan-t5-small"
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=-1,  # force CPU
        max_length=256,
    )
    return pipe

pipe = load_model()

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere echte Gerichtsurteile..."):
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
            result = pipe(prompt)
            st.write("🛠️ Raw Model Output:", result)
            if result and isinstance(result, list):
                antwort = result[0].get('generated_text') or result[0].get('text')
                st.success("📜 Prognose & Begründung:")
                st.write(antwort)
