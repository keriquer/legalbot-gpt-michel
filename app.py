import streamlit as st
import requests
from transformers import pipeline

st.set_page_config(page_title="LegalBot mit deutschem Recht", page_icon="⚖️")

st.title("⚖️ LegalBot mit deutschem Recht & Gerichtsurteilen")

# --- Fetch recent court cases from OpenLegalData.io ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=10"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'fristlose Kündigung ohne Abmahnung')")

@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "google/flan-t5-small"
    return pipeline(
        "text2text-generation",
        model=model_name,
        device=-1,  # CPU
        max_length=256,
    )

pipe = load_model()

def truncate_text(text, max_tokens):
    words = text.split()
    return " ".join(words[:max_tokens])

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere Gesetzestexte & Gerichtsurteile..."):
        # --- Find relevant court cases by keyword
        matching = [
            c for c in cases
            if 'text' in c and any(word.lower() in c['text'].lower() for word in frage.split())
        ]
        if not matching:
            st.warning("Keine relevanten Entscheidungen gefunden.")
        else:
            # Use only up to 3 relevant decisions for context
            kontext = "\n\n".join(
                [f"- {c.get('court','?')} ({c.get('date_decided','?')}): {c['text'][:200]}..." for c in matching[:3]]
            )
            kontext_trunc = truncate_text(kontext, max_tokens=400)
            frage_trunc = truncate_text(frage, max_tokens=40)
            prompt = (
                f"Du bist ein hilfreicher Rechtsassistent. "
                f"Basierend auf diesen Gerichtsurteilen:\n\n{kontext_trunc}\n\n"
                f"Beantworte folgende Frage:\n{frage_trunc}"
            )
            st.write("🛠️ DEBUG Prompt sent to model:", prompt)
            try:
                result = pipe(prompt)
            except Exception as e:
                st.error(f"Fehler beim Modellaufruf: {e}")
                result = None

            st.write("🛠️ Raw Model Output:", result)
            if result and isinstance(result, list) and ('generated_text' in result[0] or 'text' in result[0]):
                antwort = result[0].get('generated_text') or result[0].get('text')
                st.success("📜 Prognose & Begründung:")
                st.write(antwort)
            else:
                st.warning("Das Modell konnte leider keine Antwort generieren.")

st.caption("🔎 Dieses Tool nutzt Open Legal Data & das FLAN-T5 Modell. Keine echte Rechtsberatung.")
