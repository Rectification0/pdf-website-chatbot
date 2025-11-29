# app.py — FIXED FOR CHROMA TENANT ERROR IN STREAMLIT CLOUD (Nov 2025)
import os
import streamlit as st
import shutil
import tempfile

# Fix Streamlit watcher (harmless)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cloud embeddings (CPU-safe, no meta crash)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Groq LLM (fast & free)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# Config
DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Fixed DB build (with allow_reset=True to skip tenant validation)
def build_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    
    # Clean old DB
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    # KEY FIX: allow_reset=True bypasses tenant/database validation in cloud
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"},  # Optional: better search
        client_settings={"allow_reset": True}  # ← THIS SOLVES THE TENANT ERROR
    )
    st.success(f"Loaded {len(splits)} chunks — ready to chat!")

def get_retriever():
    # Load with same settings
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        client_settings={"allow_reset": True}
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})

# RAG chain
def ask_question(question):
    retriever = get_retriever()
    template = """Answer only using the following context:\n\n{context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

# PDF loader
def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs

# URL scraper
def load_url(url):
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers, timeout=20).text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return []

# UI
st.set_page_config(page_title="PDF & Website Chatbot", page_icon="🤖")
st.title("PDF + Website AI Chatbot")
st.caption("Upload a PDF or paste any public URL — ask anything!")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Load PDF", type="primary"):
        with st.spinner("Processing PDF..."):
            docs = load_pdf(uploaded_file)
            build_db(docs)

with col2:
    url = st.text_input("Or paste a public URL")
    if url and st.button("Load Website", type="primary"):
        with st.spinner("Scraping website..."):
            docs = load_url(url)
            build_db(docs)

# Chat
if os.path.exists(DB_PATH):
    st.success("Knowledge base loaded — ask questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("↑ Load a PDF or website first")

# Clear
if st.sidebar.button("Clear database (start fresh)"):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()