import streamlit as st
from oldp_client import Configuration, ApiClient
from oldp_client.api.cases_api import CasesApi
import json
from transformers import pipeline

# --- Setup Open Legal Data API client ---
config = Configuration()
# anonymous access (100 req/day). Use st.secrets for authenticated usage.
api_client = ApiClient(configuration=config)
cases_api = CasesApi(api_client)

# --- Fetch recent decisions (e.g., top 5) ---
resp = cases_api.cases_list(limit=5)
cases = resp.results  # list of case summaries with 'text' and 'court', 'date'

# --- Prepare LLM pipeline ---
pipe = pipeline(
    "text2text-generation",
    model="declare-lab/flan-alpaca-base",
    max_length=512,
    temperature=0.0
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- UI ---
st.title("⚖️ LegalBot with Open Legal Data")

frage = st.text_area("📝 Describe your German legal case")

if st.button("🔍 Get Prediction"):
    with st.spinner("Retrieving and analyzing real court decisions..."):
        # Filter decisions containing keywords
        matching = [
            c for c in cases
            if any(word.lower() in c.text.lower() for word in frage.split())
        ]
        if not matching:
            st.warning("No relevant decisions found.")
        else:
            kontext = "\n\n".join(
                [f"- {c.court} ({c.date}): {c.text[:200]}..." for c in matching]
            )
            prompt = f"You are a legal assistant. Based on these court decisions:\n\n{kontext}\n\nAnswer this legal question:\n{frage}"
            antwort = llm(prompt)
            st.success("📜 Prediction & Reasoning:")
            st.write(antwort)
