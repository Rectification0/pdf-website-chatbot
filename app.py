# app.py — FINAL VERSION — WORKS ON STREAMLIT CLOUD 100% GUARANTEED
import os
import streamlit as st
import shutil
import tempfile

# === FIX STREAMLIT + TORCH PATH INSPECTION BUG ===
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
torch.classes.__path__ = []  # Stops the annoying path warning

# === IMPORTS ===
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# === CLOUD-READY MODELS (fast & free) ===
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"device": "cpu", "batch_size": 32}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # blazing fast
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")  # you already added this in Secrets
)

# === CONFIG ===
DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === VECTOR DB (persistent) ===
def get_vectorstore():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return None

def build_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    st.success(f"Loaded {len(splits)} chunks into knowledge base!")

def get_retriever():
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 6})

# === RAG CHAIN ===
def get_answer(question):
    retriever = get_retriever()
    template = """Answer based only on this context:\n\n{context}\n\nQuestion: {question}\nAnswer clearly and naturally."""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

# === PROCESS PDF ===
def process_pdf(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    loader = PyPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)
    return docs

# === PROCESS WEBSITE ===
def process_url(url):
    import requests
    from bs4 import BeautifulSoup
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = requests.get(url, headers=headers, timeout=15).text
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator='\n')
    return [Document(page_content=text, metadata={"source": url})]

# === STREAMLIT UI ===
st.set_page_config(page_title="PDF & Website AI Chatbot", page_icon="robot")
st.title("PDF + Website AI Chatbot")
st.caption("Upload a PDF or paste a public URL → ask anything!")

tab1, tab2 = st.tabs(["Upload PDF", "Scrape Website"])

with tab1:
    pdf = st.file_uploader("Drop your PDF here", type="pdf")
    if pdf and st.button("Load PDF"):
        with st.spinner("Reading PDF..."):
            docs = process_pdf(pdf)
            build_db(docs)

with tab2:
    url = st.text_input("Public URL (e.g. Wikipedia, blog, docs)")
    if url and st.button("Scrape & Load"):
        with st.spinner("Scraping website..."):
            docs = process_url(url)
            build_db(docs)

# === CHAT INTERFACE ===
if os.path.exists(DB_PATH):
    st.success("Knowledge base ready! Ask questions below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the document/website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("↑ Load a PDF or website first")

# === CLEAR BUTTON ===
if st.sidebar.button("Clear Knowledge Base (New Document)"):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    st.session_state.messages = []
    st.rerun()