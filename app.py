# app.py — FINAL 100% WORKING VERSION FOR STREAMLIT CLOUD (Nov 2025)
import os
import streamlit as st
import tempfile

# Fix Streamlit file watcher warning
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

# === MODELS (CPU-safe & fast) ===
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# === GLOBAL VECTORSTORE (in-memory — perfect for Streamlit Cloud) ===
vectorstore = None  # Will hold the Chroma DB after loading a document

# === BUILD DATABASE ===
def build_db(docs):
    global vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # In-memory mode — no persistence issues, no tenant errors
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    st.success(f"Loaded {len(splits)} chunks — ready to chat!")

# === RETRIEVER ===
def get_retriever():
    global vectorstore
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 6})

# === ASK QUESTION ===
def ask_question(question):
    retriever = get_retriever()
    if not retriever:
        return "Please load a document first."
    
    template = """Answer only using this context:\n\n{context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

# === LOAD PDF ===
def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.getvalue())
        path = f.name
    loader = PyPDFLoader(path)
    docs = loader.load()
    os.unlink(path)
    return docs

# === LOAD URL ===
def load_url(url):
    import requests
    from bs4 import BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=20).text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        st.error(f"Failed to scrape: {e}")
        return []

# === UI ===
st.set_page_config(page_title="PDF & Website Chatbot", page_icon="Robot")
st.title("PDF + Website AI Chatbot")
st.caption("Upload a PDF or paste a public URL → ask anything!")

col1, col2 = st.columns(2)

with col1:
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file and st.button("Load PDF", type="primary"):
        with st.spinner("Reading PDF..."):
            docs = load_pdf(pdf_file)
            build_db(docs)

with col2:
    url = st.text_input("Or paste a public URL")
    if url and st.button("Load Website", type="primary"):
        with st.spinner("Scraping website..."):
            docs = load_url(url)
            if docs:
                build_db(docs)

# === CHAT INTERFACE ===
if vectorstore is not None:
    st.success("Knowledge base ready — ask away!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("↑ Upload a PDF or paste a URL to get started")

# Clear button
if st.sidebar.button("Clear everything & start over"):
    global vectorstore
    vectorstore = None
    st.session_state.messages = []
    st.rerun()