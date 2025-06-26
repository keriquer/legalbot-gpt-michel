import streamlit as st
import requests
from transformers import pipeline

API_URL = "https://de.openlegaldata.io/api/cases/?limit=5"
response = requests.get(API_URL)
if response.status_code == 200:
    cases = response.json().get('results', [])
else:
    cases = []

pipe = pipeline(
    "text2text-generation",
    model="declare-lab/flan-alpaca-base",
    max_length=512,
    temperature=0.0
)

st.title("⚖️ LegalBot mit Open Legal Data (deutsch)")
frage = st.text_area("📝 Beschreibe deinen Fall auf Deutsch")

if st.button("🔍 Prognose abrufen"):
    with st.spinner("Analysiere echte Gerichtsurteile..."):
        matching = [
            c for c in cases
            if 'text' in c and any(word.lower() in c['text'].lower() for word in frage.split())
        ]
