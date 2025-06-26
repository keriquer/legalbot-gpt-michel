import streamlit as st
import requests
import json
from transformers import pipeline

# --- Load BGB statutes ---
try:
    with open("bgb.json", "r", encoding="utf-8") as f:
        bgb_list = json.load(f)
except Exception:
    bgb_list = []

# --- Load local court decisions ---
try:
    with open("urteile.json", "r", encoding="utf-8") as f:
        local_urteile = json.load(f)
except Exception:
    local_urteile = []

# --- Fetch recent court cases from OpenLegalData.io ---
API_URL = "https://de.openlegaldata.io/api/cases/?limit=10"
response = requests.get(API_URL)
cases = response.json().get('results', []) if response.status_code == 200 else []

st.title("⚖️ LegalBot mit deutschem Recht & Gerichtsurteilen")
frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch (z.B. 'fristlose Kündigung ohne Abmahnung')")

@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        max_length=512,
    )

pipe = load_model()

def find_relevant_bgb(frage):
    # Simple keyword check. Expand for more robust matching!
    return [
        para for para in bgb_list
        if any(word.lower() in (para["text"] + para["titel"]).lower() for word in frage.split())
    ]

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere Gesetzestexte & Gerichtsurteile..."):
        # 1. Find relevant BGB paragraphs
        relevante_bgb = find_relevant_bgb(frage)
        if relevante_bgb:
            gesetz_kontext = "\n".join(
                [f'{b["paragraf"]} – {b["titel"]}: {b["text"]}' for b in relevante_bgb]
            )
        else:
            gesetz_kontext = "\n".join(
                [f'{b["paragraf"]} – {b["titel"]}: {b["text"]}' for b in bgb_list[:2]]
            )
        # 2. Try to match from API cases
        matching_api = [
            c for c in cases
            if 'text' in c and any(word.lower() in c['text'].lower() for word in frage.split())
        ]
        # 3. Try to match from local urteile
        matching_local = [
            c for c in local_urteile
            if 'inhalt' in c and any(word.lower() in c['inhalt'].lower() for word in frage.split())
        ]
        # 4. Fallback: use all local urteile
        if not matching_local:
            matching_local = local_urteile

        urteil_kontext = ""
        if matching_api:
            urteil_kontext = "\n".join(
                [f"- {c['court']} ({c['date_decided']}): {c['text'][:200]}..." for c in matching_api[:2]]
            )
        if not urteil_kontext and matching_local:
            urteil_kontext = "\n".join(
                [f"- {c['gericht']} ({c['datum']}): {c['inhalt'][:200]}..." for c in matching_local[:2]]
            )
        if not urteil_kontext:
            urteil_kontext = "Keine Gerichtsurteile gefunden."

        # -- Compose full context
        kontext = f"{gesetz_kontext}\n\n{urteil_kontext}"

        prompt = f"""
Du bist ein deutscher Rechtsgutachter. Analysiere die folgende Frage im Gutachtenstil (Obersatz, Definition, Subsumtion, Ergebnis).
Nutze die folgenden Gesetzestexte und Gerichtsurteile:
{kontext}

Frage:
{frage}

Bitte nenne, wenn möglich, die einschlägigen Paragraphen (z.B. BGB) und relevante Gerichtsentscheidungen.
"""

        result = pipe(prompt)
        st.write("🛠️ Raw Model Output:", result)
        if result and isinstance(result, list):
            antwort = result[0].get('generated_text') or result[0].get('text')
            st.success("📜 Prognose & Begründung:")
            st.write(antwort)
        else:
            st.warning("Das Modell hat keine Antwort generiert.")
