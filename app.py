import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# 🔐 Insert your API keys here
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "LegalBot-Web"

# 📁 Load your example court cases
loader = JSONLoader("urteile.json", text_content_key="inhalt")
docs = loader.load()
chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
db = FAISS.from_documents(chunks, OpenAIEmbeddings())

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🌐 Web interface
st.title("⚖️ LegalBot: Judgment Prediction")
frage = st.text_area("📋 Describe your legal case (e.g. dismissal, rent conflict, etc.):")

if st.button("🔍 Show Prediction"):
    with st.spinner("Analyzing court decisions..."):
        antwort = qa_chain.run(frage)
        st.success("📜 Prediction & Legal Reasoning:")
        st.write(antwort)
