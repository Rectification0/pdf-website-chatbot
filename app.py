import os
# Cloud mode: use Groq + HuggingFace (works everywhere)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
# —————————————————————————————————————————————————————————————
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import os
from dotenv import load_dotenv
import shutil
import tempfile

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"          # change to llama3.2:3b or llama3.2:8b if you want
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_PATH = "chroma_db"



# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def get_vectorstore():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        return None

def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    
    loader = PyPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # delete temp file
    return docs

def process_url(url):
    try:
        # Try Firecrawl first (best results)
        if os.getenv("FIRECRAWL_API_KEY"):
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
            data = app.scrape_url(url, params={'pageOptions': {'onlyMainContent': True}})
            return [Document(page_content=data['markdown'], metadata={"source": url})]
    except:
        pass
    
    # Fallback to simple requests + bs4
    import requests
    from bs4 import BeautifulSoup
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = requests.get(url, headers=headers, timeout=10).text
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    text = soup.get_text(separator='\n')
    return [Document(page_content=text, metadata={"source": url})]

def build_or_update_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)  # fresh start each time (simple mode)
    
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    st.success(f"Loaded {len(splits)} chunks! Ready to chat.")

def get_retriever():
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 6})

# RAG chain
def get_answer(question):
    retriever = get_retriever()
    template = """Answer the question using ONLY the following context. 
    If you don't know, say "I don't know".

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF & Website Chatbot", page_icon="robot")
st.title("PDF + Website AI Chatbot (Windows Version)")
st.caption("Upload a PDF or paste any public URL → ask questions!")

tab1, tab2 = st.tabs(["Upload PDF", "Scrape Website"])

with tab1:
    pdf_file = st.file_uploader("Drop your PDF here", type="pdf")
    if pdf_file and st.button("Load PDF → Ask Questions"):
        with st.spinner("Reading PDF..."):
            docs = process_pdf(pdf_file)
            build_or_update_db(docs)

with tab2:
    url = st.text_input("Public website URL (e.g. https://en.wikipedia.org/wiki/Python)")
    if url and st.button("Scrape Website → Ask Questions"):
        with st.spinner("Scraping website..."):
            docs = process_url(url)
            build_or_update_db(docs)

# Chat
if os.path.exists(DB_PATH):
    st.success("Knowledge base ready!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ask anything about the document or website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("↑ First load a PDF or website")

# Sidebar clear button
if st.sidebar.button("Clear Everything (New Document)"):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    st.session_state.messages = []
    st.rerun()