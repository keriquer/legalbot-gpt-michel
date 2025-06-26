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

# --- Use a small, CPU-friendly model for maximum compatibility ---
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",   # <- lightweight and cloud-compatible!
    max_length=256,
    device=-1                       # <- force CPU, works on Streamlit Cloud
)

st.title("⚖️ LegalBot mit Open Legal Data (deutsch)")
frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'Kündigung')")

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere echte Gerichtsurteile..."):
        # Simple keyword matching against fetched court case texts
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
            st.write("🛠️ Raw Model Output:", result)  # Debug: See what the model returns
            if result and isinstance(result, list):
                antwort = result[0].get('generated_text') or result[0].get('text')
                st.success("📜 Prognose & Begründung:")
