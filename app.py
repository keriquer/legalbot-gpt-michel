import streamlit as st
import requests
from transformers import pipeline

st.set_page_config(page_title="LegalBot mit deutschem Recht", page_icon="⚖️")
st.title("⚖️ LegalBot mit deutschem Recht & Gerichtsurteilen")

frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'fristlose Kündigung ohne Abmahnung')")

@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "google/flan-t5-small"
    return pipeline(
        "text2text-generation",
        model=model_name,
        device=-1,
        max_length=256,
    )

pipe = load_model()

if st.button("🔍 Prognose abrufen") and frage.strip():
    with st.spinner("Analysiere Gesetzestexte & Gerichtsurteile..."):
        # Try to find a good search term
        words = frage.lower().split()
        # Try to pick a likely legal keyword (prefer 'kündigung', etc. else just first word)
        for keyword in ["kündigung", "abmahnung", "arbeitsrecht", "sozialauswahl", "behinderung"]:
            if keyword in words:
                search_term = keyword
                break
        else:
            search_term = words[0] if words else "recht"
        # Fetch cases
        API_URL = f"https://de.openlegaldata.io/api/cases/?q={search_term}"
        response = requests.get(API_URL)
        if response.status_code == 200:
            cases = response.json().get('results', [])
        else:
            cases = []
        # Now match more specifically
        matching = [
            c for c in cases
            if 'text' in c and any(w in c['text'].lower() for w in words)
        ]
        if not matching:
            st.warning("Keine relevanten Entscheidungen gefunden.")
        else:
            kontext = "\n\n".join(
                [f"- {c.get('court', 'Unbekannt')} ({c.get('date_decided', 'kein Datum')}): {c['text'][:200]}..." for c in matching[:3]]
            )
            prompt = f"Du bist ein hilfreicher Rechtsassistent. Basierend auf diesen Gerichtsurteilen:\n\n{kontext}\n\nBeantworte folgende Frage:\n{frage}"
            result = pipe(prompt)
            st.write("🛠️ Raw Model Output:", result)
            if result and isinstance(result, list):
                antwort = result[0].get('generated_text') or result[0].get('text')
                st.success("📜 Prognose & Begründung:")
                st.write(antwort)
