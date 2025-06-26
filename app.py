import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# 🔐 Set your API keys (replace these!)
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "LegalBot-Web"

# 📄 Load your legal data
loader = JSONLoader(
    file_path="urteile.json",
    jq_schema=".[]",
    text_content_key="inhalt"
)
docs = loader.load()

# 🧠 Prepare embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
db = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = db.as_retriever(search_kwargs={"k": 3})

# 🔍 Retrieval QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🌐 Streamlit UI
st.title("⚖️ LegalBot – GPT-powered Judgment Predictor")
frage = st.text_area("📋 Describe your legal case here:")

if st.button("🔍 Predict Outcome"):
    with st.spinner("Analyzing court decisions..."):
        antwort = qa_chain.run(frage)
        st.success("📜 Prediction & Legal Reasoning:")
        st.write(antwort)
