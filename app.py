import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Import backend logic
try:
    from backend import (
        logger,
        scrape_website,
        create_vector_db,
        get_llm,
        extract_llm_response,
    )
except ImportError as e:
    st.error(f"Missing backend dependency: {e}")
    st.stop()
    

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Page Config ---
st.set_page_config(page_title="AI Document & Web Chatbot", layout="wide")
st.title("🤖 AI-Powered Document & Web Chatbot")

# Check imports and show helpful error messages
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Please run: pip install -r requirements.txt")
    st.stop()


# Initialize logger
logger.log_event("APP_START", {"message": "Application started"})


# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")

# --- FIXED: Updated to currently available Groq models (as of Dec 2024) ---
model_options = {
    "llama-3.3-70b-versatile": "🚀 Llama 3.3 70B - Best overall (Recommended)",
    "llama-3.1-8b-instant": "💨 Llama 3.1 8B - Fast responses",
    "mixtral-8x7b-32768": "📚 Mixtral 8x7B - Large context",
}

model_name = st.sidebar.selectbox(
    "Choose AI Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0,
)

temperature = st.sidebar.slider(
    "Temperature (creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Lower = more focused, Higher = more creative",
)

max_tokens = st.sidebar.slider(
    "Max response length",
    min_value=512,
    max_value=4096,
    value=2048,
    step=256,
    help="Maximum length of AI responses",
)

# Log summary in sidebar
st.sidebar.divider()
st.sidebar.subheader("📊 Activity Log")
log_summary = logger.get_log_summary()
if log_summary:
    st.sidebar.caption(f"🖥️ Deployment: {log_summary['deployment']}")
    st.sidebar.caption(f"📝 Total Events: {log_summary['total_events']}")
    st.sidebar.caption(f"❌ Total Errors: {log_summary['total_errors']}")
    st.sidebar.caption(f"🕐 Started: {log_summary['created_at'][:10]}")


# --- Main Layout: Two Columns ---
col1, col2 = st.columns(2)

# ========== LEFT COLUMN: PDF CHATBOT ==========
with col1:
    st.header("📄 PDF Chatbot")
    st.caption("Upload a PDF and ask questions about its content")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

    if uploaded_file:
        if st.button("🚀 Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    logger.log_event("PDF_UPLOAD", {"filename": uploaded_file.name})

                    # Clear old vector database
                    if "pdf_vectordb" in st.session_state:
                        try:
                            del st.session_state.pdf_vectordb
                            logger.log_event(
                                "VECTORDB_CLEARED",
                                {"message": "Old vector database cleared"},
                            )
                        except Exception as e:
                            logger.log_error("VECTORDB_CLEAR_ERROR", str(e))

                    # Clear chat history
                    st.session_state.pdf_messages = []
                    st.session_state.pdf_processed = False

                    # Save PDF to temp file
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_pdf.write(uploaded_file.read())
                    temp_pdf.close()

                    # Load and process PDF
                    loader = PyPDFLoader(temp_pdf.name)
                    docs = loader.load()

                    if not docs:
                        logger.log_error(
                            "PDF_PROCESSING_ERROR",
                            "No text extracted",
                            {"filename": uploaded_file.name},
                        )
                        st.error(
                            "Could not extract text from PDF. Please try another file."
                        )
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500, chunk_overlap=50
                        )  # Smaller chunks
                        splits = text_splitter.split_documents(docs)

                        vectordb = create_vector_db(splits)
                        if vectordb:
                            st.session_state.pdf_vectordb = vectordb
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_name = uploaded_file.name
                            logger.log_pdf_processing(
                                uploaded_file.name, len(splits), success=True
                            )
                            st.success(
                                f"✅ PDF processed! {len(splits)} chunks indexed from '{uploaded_file.name}'"
                            )
                            st.info(
                                "💬 Previous chat history cleared. Ready for new questions!"
                            )

                    # Clean up temp file
                    try:
                        os.unlink(temp_pdf.name)
                    except Exception:
                        pass

                except Exception as e:
                    logger.log_error(
                        "PDF_PROCESSING_ERROR", str(e), {"filename": uploaded_file.name}
                    )
                    st.error(f"Error processing PDF: {e}")

    st.divider()

    # PDF Chat Interface
    if "pdf_messages" not in st.session_state:
        st.session_state.pdf_messages = []

    pdf_chat_container = st.container()
    with pdf_chat_container:
        for message in st.session_state.pdf_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.get("pdf_processed"):
        st.caption(f"💬 Chatting about: {st.session_state.get('pdf_name', 'PDF')}")
        if pdf_prompt := st.chat_input("Ask about the PDF...", key="pdf_chat"):
            st.session_state.pdf_messages.append(
                {"role": "user", "content": pdf_prompt}
            )

            with pdf_chat_container:
                with st.chat_message("user"):
                    st.markdown(pdf_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing PDF..."):
                        try:
                            vectordb = st.session_state.pdf_vectordb
                            retriever = vectordb.as_retriever(search_kwargs={"k": 4})

                            # Retrieve relevant documents
                            try:
                                pdf_docs = retriever.invoke(pdf_prompt)
                            except Exception as retr_err:
                                logger.log_error("RETRIEVAL_ERROR", str(retr_err))
                                pdf_docs = []

                            pdf_context = (
                                "\n\n".join(
                                    [
                                        getattr(d, "page_content", str(d))
                                        for d in pdf_docs
                                    ]
                                )
                                if pdf_docs
                                else ""
                            )

                            llm = get_llm(model_name, temperature, max_tokens)
                            if llm is None:
                                error_msg = "⚠️ PDF mode requires a Groq model and valid GROQ_API_KEY. Please configure in Settings or .env file."
                                st.error(error_msg)
                                st.session_state.pdf_messages.append(
                                    {"role": "assistant", "content": error_msg}
                                )
                                logger.log_event(
                                    "PDF_ANSWER_SKIPPED", {"model_selected": model_name}
                                )
                                st.rerun()

                            # Create prompt
                            answer_prompt = """You are an expert AI assistant analyzing a PDF document. You have NO access to the internet or external information.

PDF Content:
{pdf_context}

STRICT INSTRUCTIONS:
- Answer ONLY using information explicitly found in the PDF content above
- DO NOT use any external knowledge, internet information, or general knowledge
- DO NOT make assumptions or inferences beyond what's in the PDF
- If the PDF doesn't contain the information needed to answer the question, you MUST respond with:
  "I don't have enough information in this PDF to answer that question. The document doesn't contain details about [topic]."
- Be honest about the limitations of the PDF content
- Provide detailed answers ONLY when the information is clearly present in the PDF
- Quote or reference specific parts of the PDF when possible
- Use a professional yet conversational tone

Question: {question}

Answer (based ONLY on the PDF content above):"""

                            prompt_template = ChatPromptTemplate.from_template(
                                answer_prompt
                            )
                            formatted_prompt = prompt_template.format(
                                pdf_context=(
                                    pdf_context
                                    if pdf_context
                                    else "No relevant content found in PDF"
                                ),
                                question=pdf_prompt,
                            )

                            # Get answer using improved extraction
                            answer = extract_llm_response(llm, formatted_prompt)

                            st.markdown(answer)
                            st.session_state.pdf_messages.append(
                                {"role": "assistant", "content": answer}
                            )
                            logger.log_chat_interaction(
                                "PDF", len(pdf_prompt), len(answer)
                            )

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            logger.log_error(
                                "PDF_CHAT_ERROR", str(e), {"question": pdf_prompt[:100]}
                            )
                            st.error(error_msg)
                            st.session_state.pdf_messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

            st.rerun()
    else:
        st.info("👆 Upload a PDF and click 'Process PDF' to start chatting!")


# ========== RIGHT COLUMN: WEBSITE CHATBOT ==========
with col2:
    st.header("🌐 Website Chatbot")
    st.caption("Enter a website URL and ask questions about its content")

    with st.expander("⚠️ Responsible Web Scraping Guidelines"):
        st.markdown(
            """
        **Please use this tool responsibly:**
        
        ✅ **Allowed:**
        - Public news websites and blogs
        - Educational and research content
        - Open documentation and wikis
        - Your own websites
        
        ❌ **Not Allowed:**
        - Banking or financial sites
        - Login/authentication pages
        - Medical or health records
        - Government/military sites
        - Paywalled content
        - Private or internal networks
        
        **Legal Notice:**
        - Only scrape publicly accessible content
        - Respect website Terms of Service
        - Use for personal/educational purposes only
        - Do not use scraped content commercially without permission
        
        By using this tool, you agree to follow these guidelines and applicable laws.
        """
        )

    website_url_input = st.text_input(
        "Enter Website URL", placeholder="https://example.com", key="website_url_input"
    )

    if website_url_input:
        if st.button("🔍 Scrape Website", type="primary", key="scrape_button"):
            with st.spinner("Validating and scraping website..."):
                result = scrape_website(website_url_input)
                if result["success"]:
                    st.session_state.website_content = result["content"]
                    st.session_state.website_title = result["title"]
                    st.session_state.scraped_url = result["url"]
                    st.session_state.website_processed = True
                    st.session_state.website_messages = []  # Clear chat history
                    st.success(f"✅ Website scraped: {result['title']}")
                    st.caption(f"Content length: {len(result['content'])} characters")
                    if result.get("warning"):
                        st.warning(f"⚠️ {result['warning']}")
                else:
                    st.error(result["error"])
                    st.session_state.website_processed = False

    st.divider()

    if "website_messages" not in st.session_state:
        st.session_state.website_messages = []

    web_chat_container = st.container()
    with web_chat_container:
        for message in st.session_state.website_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.get("website_processed"):
        st.caption(
            f"💬 Chatting about: {st.session_state.get('website_title', 'Website')}"
        )
        if web_prompt := st.chat_input("Ask about the website...", key="web_chat"):
            st.session_state.website_messages.append(
                {"role": "user", "content": web_prompt}
            )

            with web_chat_container:
                with st.chat_message("user"):
                    st.markdown(web_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing website..."):
                        try:
                            website_content = st.session_state.website_content
                            website_title = st.session_state.website_title
                            website_url = st.session_state.scraped_url

                            # Truncate if too long
                            max_content_length = 15000
                            if len(website_content) > max_content_length:
                                website_content = (
                                    website_content[:max_content_length] + "..."
                                )

                            llm = get_llm(model_name, temperature, max_tokens)
                            if llm is None:
                                error_msg = "⚠️ Website mode requires a Groq model and valid GROQ_API_KEY. Please configure in Settings or .env file."
                                st.error(error_msg)
                                st.session_state.website_messages.append(
                                    {"role": "assistant", "content": error_msg}
                                )
                                logger.log_event(
                                    "WEBSITE_ANSWER_SKIPPED",
                                    {"model_selected": model_name},
                                )
                                st.rerun()

                            # Create prompt
                            answer_prompt = """You are an expert AI assistant analyzing website content.

Website: {website_title}
URL: {website_url}

Website Content:
{website_content}

Instructions:
- Answer the question using ONLY the information from the website
- Provide detailed, well-structured answers
- Use your reasoning to extract relevant information
- If the website doesn't contain the answer, say so honestly
- Be clear, accurate, and helpful
- Use a professional yet conversational tone

Question: {question}

Answer:"""

                            prompt_template = ChatPromptTemplate.from_template(
                                answer_prompt
                            )
                            formatted_prompt = prompt_template.format(
                                website_title=website_title,
                                website_url=website_url,
                                website_content=website_content,
                                question=web_prompt,
                            )

                            # Get answer using improved extraction
                            answer = extract_llm_response(llm, formatted_prompt)

                            st.markdown(answer)
                            st.session_state.website_messages.append(
                                {"role": "assistant", "content": answer}
                            )
                            logger.log_chat_interaction(
                                "WEBSITE", len(web_prompt), len(answer)
                            )

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            logger.log_error(
                                "WEBSITE_CHAT_ERROR",
                                str(e),
                                {"question": web_prompt[:100]},
                            )
                            st.error(error_msg)
                            st.session_state.website_messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

            st.rerun()
    else:
        st.info("👆 Enter a website URL and click 'Scrape Website' to start chatting!")


# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("💡 Tip: Use both modes simultaneously!")
