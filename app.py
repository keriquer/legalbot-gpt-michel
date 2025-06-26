import streamlit as st
import os
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# 🔐 Load API keys securely from Streamlit Cloud secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# 📄 Load your legal cases from urteile.json
with open("urteile.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🧾 Convert JSON entries into LangChain Documents
docs = [Document(page_content=item["inhalt"], metadata=item) for item in data]

# 🔍 Split and embed the documents
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 🤖 GPT-4 QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🌐 Streamlit Web App UI
st.set_page_config(page_title="LegalBot", page_icon="⚖️")
st.title("⚖️ LegalBot – GPT-powered Judgment Predictor")
st.write("Enter a legal case description (e.g. employment dismissal, rent termination):")

frage = st.text_area("📝 Case Description")

if st.button("🔍 Get Prediction"):
    with st.spinner("Analyzing legal precedents..."):
        antwort = qa_chain.run(frage)
        st.success("📜 Prediction & Legal Reasoning")
        st.write(antwort)
