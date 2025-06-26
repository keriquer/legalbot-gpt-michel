import streamlit as st
import requests
import json
from transformers import pipeline

# --- Fetch recent court cases from OpenLegalData.io ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=10"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

# --- Load your own fallback urteile.json for guaranteed context ---
try:
    with open("urteile.json", "r", encoding="utf-8") as f:
        local_urteile = json.load(f)
except Exception as e:
    local_urteile = []

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
        # 1. Try to match from online cases
        matching = [
            c for c in cases
            if 'text' in c and any(word.lower() in c['text'].lower() for word in frage.split())
        ]
        # 2. If not found, use local urteile.json
        if not matching and local_urteile:
            matching = [
                c for c in local_urteile
                if 'inhalt' in c and any(word.lower() in c['inhalt'].lower() for word in frage.split())
            ]
            # If still no matches, use ALL local urteile as context (so it always finds something)
            if not matching:
                matching = local_urteile
            # Prepare context from local urteile
            kontext = "\n\n".join(
                [f"- {c['gericht']} ({c['datum']}): {c['inhalt'][:200]}..." for c in matching[:3]]
            )
        else:
            # Prepare context from API cases
            kontext = "\n\n".join(
                [f"- {c['court']} ({c['date_decided']}): {c['text'][:200]}..." for c in matching[:3]]
            )

        # If everything failed (no API, no local), fall back to a general legal answer
        if not matching and not local_urteile:
            kontext = "Es liegen keine konkreten Urteile vor. Bitte beachte das deutsche Recht und übliche gerichtliche Praxis."

        # 3. Build prompt and run AI
        prompt = f"Du bist ein hilfreicher Rechtsassistent. Basierend auf diesen Gerichtsurteilen:\n\n{kontext}\n\nBeantworte folgende Frage:\n{frage}"
        result = pipe(prompt)
        st.write("🛠️ Raw Model Output:", result)
        if result and isinstance(result, list):
            antwort = result[0].get('generated_text') or result[0].get('text')
            st.success("📜 Prognose & Begründung:")
            st.write(antwort)
        else:
            st.warning("Das Modell hat keine Antwort generiert.")
