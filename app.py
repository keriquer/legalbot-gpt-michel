import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import os
import json

# 🔐 API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "LegalBot-Web"

# 📄 Load urteile.json manually and wrap as Document objects
with open("urteile.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = [Document(page_content=item["inhalt"], metadata=item) for item in data]

# 🧠 Split and embed
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 🔍 Retrieval QA
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🌐 Streamlit UI
st.title("⚖️ LegalBot – GPT-powered Judgment Predictor")
frage = st.text_area("📋 Describe your legal case here:")

if st.button("🔍 Predict Outcome"):
    with st.spinner("Analyzing similar court cases..."):
        antwort = qa_chain.run(frage)
        st.success("📜 Prediction & Legal Reasoning:")
        st.write(antwort)
