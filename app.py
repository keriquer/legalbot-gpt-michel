import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 🔐 Secure key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 🧠 Load court decisions
with open("urteile.json", "r", encoding="utf-8") as f:
    urteile = json.load(f)

# 📄 Prepare prompt template
template = """You are a legal assistant. Based on the following court decisions:

{kontext}

Now answer this legal question:

{frage}

Only refer to what is stated in the provided decisions.
"""

prompt = PromptTemplate(input_variables=["kontext", "frage"], template=template)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# 🌐 UI
st.set_page_config(page_title="LegalBot Lite", page_icon="⚖️")
st.title("⚖️ LegalBot – Lite Version (No Embedding)")
frage = st.text_area("📝 Describe your legal issue")

if st.button("🔍 Get Prediction"):
    with st.spinner("Checking relevant decisions..."):
        # Simple matching: include all cases that mention key words
        matching = [u for u in urteile if any(word.lower() in u["inhalt"].lower() for word in frage.split())]
        kontext = "\n\n".join([f"- {u['gericht']} ({u['datum']}): {u['inhalt']}" for u in matching[:3]])

        if not kontext:
            st.warning("No relevant decisions found.")
        else:
            antwort = chain.run(kontext=kontext, frage=frage)
            st.success("📜 Prediction & Legal Explanation:")
            st.write(antwort)
