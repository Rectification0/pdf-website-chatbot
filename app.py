import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tempfile

# --- Page Config ---
st.set_page_config(page_title="PDF RAG with Groq", layout="centered")
st.title("Chat with your PDF using Groq (Lightning Fast!)")

# Model selection
model_name = st.selectbox(
    "Choose Groq model",
    [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-11b-vision-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    index=0,
)

@st.cache_resource
def load_chain(model: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    persist_dir = tempfile.mkdtemp()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
        st.stop()

    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.7)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, just say you don't know. "
        "Keep the answer concise (max 3 sentences)."
        "\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, vectordb


# --- PDF Upload & Processing ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(uploaded_file.read())
        temp_pdf.close()

        loader = PyPDFLoader(temp_pdf.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        chain, vectordb = load_chain(model_name)
        vectordb.delete_collection()
        vectordb.add_documents(splits)

        st.session_state.chain = chain
        st.session_state.vectordb = vectordb
        st.success(f"PDF processed! {len(splits)} chunks indexed.")


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "chain" in st.session_state:
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF and click 'Process PDF' to begin!")