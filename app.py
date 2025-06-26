import streamlit as st
import json
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load German court decisions from JSON
with open("urteile.json", "r", encoding="utf-8") as f:
    urteile = json.load(f)

# Prompt template
template = """Du bist ein hilfreicher rechtlicher Assistent. Basierend auf den folgenden Gerichtsurteilen:

{kontext}

Beantworte bitte die Rechtsfrage:

{frage}

Nutze nur Fakten aus den obigen Urteilen und antworte auf Deutsch."""
prompt = PromptTemplate(input_variables=["kontext", "frage"], template=template)

# Load a free model via Hugging Face
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512,
    temperature=0.0
)
llm = HuggingFacePipeline(pipeline=pipe)

# Build chain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.set_page_config(page_title="LegalBot ‑ Free", page_icon="⚖️")
st.title("⚖️ LegalBot – kostenlose Version 🇩🇪")

frage = st.text_area("📝 Beschreibe dein rechtliches Anliegen auf Deutsch:")

if st.button("🔍 Urteil finden & analysieren"):
    if not frage.strip():
        st.warning("Bitte gib deine Frage ein.")
    else:
        # Simple keyword-based matching
        keywords = frage.lower().split()
        matches = [u for u in urteile if any(kw in u["inhalt"].lower() for kw in keywords)]
        kontext = "\n\n".join(f"- {u['gericht']} ({u['datum']}): {u['inhalt']}" for u in matches[:3])

        if not kontext:
            st.warning("⚠️ Keine passenden Gerichtsurteile gefunden.")
        else:
            with st.spinner("Analysiere Urteile..."):
                antwort = chain.invoke({"kontext": kontext, "frage": frage})
            st.markdown("### 📜 Gefundene Urteile")
            st.write(kontext)
            st.markdown("### 🧠 Einschätzung")
            st.write(antwort)
