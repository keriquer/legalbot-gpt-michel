import streamlit as st
import requests
import re
from transformers import pipeline

st.title("⚖️ LegalBot mit deutschem Recht & Gerichtsurteilen")

frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch...")

@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        max_length=256,
    )

pipe = load_model()

def fetch_cases(frage):
    # Extract keywords (German words)
    frage_keywords = re.findall(r"\w+", frage.lower())
    found_cases = []
    for keyword in frage_keywords[:3]:  # limit to 3 keywords
        response = requests.get(f"https://de.openlegaldata.io/api/cases/?q={keyword}")
        if response.status_code == 200:
            found_cases.extend(response.json().get("results", []))
    # Remove duplicates by case ID
    unique_cases = {c["id"]: c for c in found_cases if "id" in c}.values()
    return list(unique_cases)

if st.button("🔍 Prognose abrufen"):
    cases = fetch_cases(frage)
    if not cases:
        st.warning("Keine relevanten Entscheidungen gefunden. Versuche andere Schlüsselwörter.")
    else:
        # Prepare the context for the AI model
        kontext = "\n\n".join(
            [f"- {c.get('court', 'Gericht unbekannt')} ({c.get('date_decided', 'Datum unbekannt')}): {c.get('text', '')[:200]}..." for c in cases if 'text' in c]
        )
        prompt = f"Du bist ein hilfreicher Rechtsassistent. Basierend auf diesen Gerichtsurteilen:\n\n{kontext}\n\nBeantworte folgende Frage:\n{frage}"
        result = pipe(prompt)
        st.write("🛠️ Raw Model Output:", result)
        if result and isinstance(result, list):
            antwort = result[0].get('generated_text') or result[0].get('text')
            st.success("📜 Prognose & Begründung:")
            st.write(antwort)
