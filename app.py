import streamlit as st
import requests
from transformers import pipeline

# --- Load recent German court cases ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=5"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

# --- Setup Hugging Face pipeline (no temperature argument) ---
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

st.title("⚖️ LegalBot mit Open Legal Data (deutsch)")
frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'Kündigung')")

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere echte Gerichtsurteile..."):
        # Very simple keyword matching
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
            # Show the raw model output for debugging
            st.write("🛠️ Raw Model Output:", result)
            # Try to get the answer from either field
            if result and isinstance(result, list):
                antwort = result[0].get('generated_text') or result[0].get('text')
                st.success("📜 Prognose & Begründung:")
                st.write(antwort)
            else:
                st.warning("Das Modell hat keine Antwort generiert.")
