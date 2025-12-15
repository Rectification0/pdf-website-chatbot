import streamlit as st
import tempfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
import json
import socket

# --- Page Config ---
st.set_page_config(page_title="AI Document & Web Chatbot", layout="wide")
st.title("🤖 AI-Powered Document & Web Chatbot")

# Check imports and show helpful error messages
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Please run: pip install -r requirements.txt")
    st.stop()


# --- Logging System ---
class ChatbotLogger:
    def __init__(self):
        self.is_local = self._is_local_deployment()
        if self.is_local:
            self.log_file = "chatbot_local_logs.json"
            self.deployment_type = "LOCAL"
        else:
            self.log_file = "chatbot_web_logs.json"
            self.deployment_type = "WEB"
        self._initialize_log_file()
    def _is_local_deployment(self):
        """Detect if running locally or on Streamlit Cloud"""
        try:
            if os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS"):
                return False
            hostname = socket.gethostname()
            if "streamlit" in hostname.lower() or "cloud" in hostname.lower():
                return False
            return True
        except:
            return True
    
    def _initialize_log_file(self):
        """Create log file if missing"""
        try:
            if not os.path.exists(self.log_file):
                initial_data = {
                    "deployment_type": self.deployment_type,
                    "created_at": datetime.now().isoformat(),
                    "logs": []
                }
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_data, f, indent=2)
        except Exception as e:
            print(f"Error initializing log file: {e}")
    
    def log_event(self, event_type, details, status="success"):
        """Append an event to the log file (best-effort)"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "deployment": self.deployment_type,
                "event_type": event_type,
                "status": status,
                "details": details
            }
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {
                    "deployment_type": self.deployment_type,
                    "created_at": datetime.now().isoformat(),
                    "logs": []
                }
            data["logs"].append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False
    
    def log_error(self, error_type, error_message, context=None):
        """Log an error entry"""
        details = {
            "error_type": error_type,
            "error_message": str(error_message),
            "context": context or {}
        }
        return self.log_event("ERROR", details, status="error")
    
    def log_pdf_processing(self, filename, chunks_count, success=True):
        """Log PDF processing event"""
        details = {"filename": filename, "chunks_count": chunks_count}
        status = "success" if success else "error"
        return self.log_event("PDF_PROCESSING", details, status=status)
    
    def log_website_scraping(self, url, content_length, success=True):
        """Log website scraping event"""
        details = {"url": url, "content_length": content_length}
        status = "success" if success else "error"
        return self.log_event("WEBSITE_SCRAPING", details, status=status)
    
    def log_chat_interaction(self, mode, question_length, answer_length):
        """Log chat interaction"""
        details = {"mode": mode, "question_length": question_length, "answer_length": answer_length}
        return self.log_event("CHAT_INTERACTION", details, status="success")
    
    def get_log_summary(self):
        """Return aggregated log summary"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total_logs = len(data.get("logs", []))
            errors = sum(1 for log in data.get("logs", []) if log.get("status") == "error")
            return {
                "total_events": total_logs,
                "total_errors": errors,
                "deployment": data.get("deployment_type", "UNKNOWN"),
                "created_at": data.get("created_at", "Unknown")
            }
        except:
            return None
# --- Cache the Embedding Model (CRITICAL FIX) ---
@st.cache_resource
def get_shared_embeddings():
    """
    Loads the embedding model ONLY ONCE and shares it across all users.
    This prevents the app from crashing due to running out of memory.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# Initialize logger
logger = ChatbotLogger()
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


# --- Helper Functions ---
def validate_url(url: str) -> dict:
    """Validate URL and check for problematic domains"""
    blocked_patterns = [
        "bank", "paypal", "stripe", "payment",
        "login", "signin", "auth", "account",
        "admin", "dashboard", "portal",
        "medical", "health", "patient",
        "gov", "military", "defense",
        "private", "internal", "intranet"
    ]
    suspicious_tlds = [".onion", ".i2p"]
    try:
        parsed = urlparse(url.lower())
        if not parsed.scheme:
            return {"valid": False, "error": "Invalid URL: Missing http:// or https://"}
        if parsed.scheme not in ["http", "https"]:
            return {"valid": False, "error": "Only HTTP and HTTPS protocols are allowed"}
        if any(x in parsed.netloc for x in ["localhost", "127.0.0.1", "0.0.0.0", "192.168.", "10."]):
            return {"valid": False, "error": "Cannot scrape local or private network addresses"}
        for pattern in blocked_patterns:
            if pattern in parsed.netloc or pattern in parsed.path:
                return {"valid": False, "error": f"This appears to be a sensitive website ({pattern}). Scraping is not allowed for security and privacy reasons."}
        for tld in suspicious_tlds:
            if parsed.netloc.endswith(tld):
                return {"valid": False, "error": "This domain type is not supported"}
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Invalid URL format: {str(e)}"}


def check_robots_txt(url: str) -> dict:
    """Check robots.txt for scraping permission"""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            robots_content = response.text.lower()
            if "disallow: /" in robots_content and "user-agent: *" in robots_content:
                return {"allowed": False, "warning": "This website's robots.txt discourages scraping. Please respect their wishes."}
        return {"allowed": True}
    except:
        return {"allowed": True, "warning": "Could not check robots.txt"}


def scrape_website(url: str) -> dict:
    """Scrape content from a website with safety checks"""
    try:
        validation = validate_url(url)
        if not validation["valid"]:
            logger.log_error("URL_VALIDATION_FAILED", validation["error"], {"url": url})
            return {"success": False, "error": validation["error"]}
        robots_check = check_robots_txt(url)
        if not robots_check.get("allowed", True):
            logger.log_error("ROBOTS_TXT_DISALLOW", robots_check["warning"], {"url": url})
            return {"success": False, "error": robots_check["warning"]}
        logger.log_event("WEBSITE_SCRAPE_START", {"url": url})
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            element.decompose()
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "No title"
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        logger.log_website_scraping(url, len(text), success=True)
        return {"success": True, "title": title_text, "content": text, "url": url, "warning": robots_check.get("warning")}
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch URL: {str(e)}"
        logger.log_error("WEBSITE_SCRAPE_ERROR", error_msg, {"url": url})
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error scraping website: {str(e)}"
        logger.log_error("WEBSITE_SCRAPE_ERROR", error_msg, {"url": url})
        return {"success": False, "error": error_msg}


def get_embeddings():
    """Return HuggingFace embeddings instance"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        logger.log_error("EMBEDDINGS_ERROR", str(e))
        raise


def create_vector_db(docs):
    try:
        logger.log_event("VECTORDB_CREATE_START", {"num_documents": len(docs)})
        
        # USE THE CACHED MODEL instead of loading a new one
        embeddings = get_shared_embeddings()
        
        # Create VectorStore (FAISS)
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        logger.log_event("VECTORDB_CREATE_SUCCESS", {"num_documents": len(docs)})
        return vectorstore
            
    except Exception as e:
        logger.log_error("VECTORDB_CREATE_ERROR", str(e), {"num_documents": len(docs)})
        st.error(f"Error creating vector database: {e}")
        return None
    
def get_llm(model: str, temp: float = 0.7, max_tok: int = 2048):
    """Return a Groq LLM instance for Groq-compatible models."""
    try:
        # Get API key from secrets or environment
        api_key = None
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key or api_key == "your-groq-api-key-here":
            logger.log_error("LLM_INIT_ERROR", "Missing or invalid API key")
            return None
        
        return ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=temp,
            max_tokens=max_tok,
            streaming=False
        )
    except Exception as e:
        logger.log_error("LLM_INIT_ERROR", str(e), {"model": model})
        return None


def extract_llm_response(llm, prompt: str) -> str:
    """
    FIXED: Robust LLM response extraction that handles ALL response types
    This fixes the "Got unknown type H/Y" errors
    """
    try:
        # Try invoke method (modern LangChain)
        response = llm.invoke(prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            # AIMessage with content attribute
            return str(response.content)
        elif isinstance(response, str):
            # Direct string response
            return response
        elif isinstance(response, dict) and 'text' in response:
            # Dict with text key
            return response['text']
        elif isinstance(response, dict) and 'content' in response:
            # Dict with content key
            return response['content']
        else:
            # Fallback: convert to string
            response_str = str(response)
            # Log the response type for debugging
            logger.log_event("LLM_RESPONSE_TYPE", {
                "type": str(type(response)),
                "preview": response_str[:100]
            })
            return response_str
            
    except Exception as e:
        logger.log_error("LLM_INVOKE_ERROR", str(e))
        return f"Error calling LLM: {str(e)}"


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
                            logger.log_event("VECTORDB_CLEARED", {"message": "Old vector database cleared"})
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
                        logger.log_error("PDF_PROCESSING_ERROR", "No text extracted", {"filename": uploaded_file.name})
                        st.error("Could not extract text from PDF. Please try another file.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Smaller chunks
                        splits = text_splitter.split_documents(docs)
                        
                        vectordb = create_vector_db(splits)
                        if vectordb:
                            st.session_state.pdf_vectordb = vectordb
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_name = uploaded_file.name
                            logger.log_pdf_processing(uploaded_file.name, len(splits), success=True)
                            st.success(f"✅ PDF processed! {len(splits)} chunks indexed from '{uploaded_file.name}'")
                            st.info("💬 Previous chat history cleared. Ready for new questions!")
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_pdf.name)
                    except:
                        pass
                        
                except Exception as e:
                    logger.log_error("PDF_PROCESSING_ERROR", str(e), {"filename": uploaded_file.name})
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
            st.session_state.pdf_messages.append({"role": "user", "content": pdf_prompt})
            
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

                            pdf_context = "\n\n".join([
                                getattr(d, "page_content", str(d)) for d in pdf_docs
                            ]) if pdf_docs else ""
                            
                            llm = get_llm(model_name, temperature, max_tokens)
                            if llm is None:
                                error_msg = "⚠️ PDF mode requires a Groq model and valid GROQ_API_KEY. Please configure in Settings or .env file."
                                st.error(error_msg)
                                st.session_state.pdf_messages.append({"role": "assistant", "content": error_msg})
                                logger.log_event("PDF_ANSWER_SKIPPED", {"model_selected": model_name})
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
                            
                            prompt_template = ChatPromptTemplate.from_template(answer_prompt)
                            formatted_prompt = prompt_template.format(
                                pdf_context=pdf_context if pdf_context else "No relevant content found in PDF",
                                question=pdf_prompt,
                            )
                            
                            # Get answer using improved extraction
                            answer = extract_llm_response(llm, formatted_prompt)
                            
                            st.markdown(answer)
                            st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
                            logger.log_chat_interaction("PDF", len(pdf_prompt), len(answer))
                            
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            logger.log_error("PDF_CHAT_ERROR", str(e), {"question": pdf_prompt[:100]})
                            st.error(error_msg)
                            st.session_state.pdf_messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
    else:
        st.info("👆 Upload a PDF and click 'Process PDF' to start chatting!")


# ========== RIGHT COLUMN: WEBSITE CHATBOT ==========
with col2:
    st.header("🌐 Website Chatbot")
    st.caption("Enter a website URL and ask questions about its content")
    
    with st.expander("⚠️ Responsible Web Scraping Guidelines"):
        st.markdown("""
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
        """)

    website_url_input = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        key="website_url_input"
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
        st.caption(f"💬 Chatting about: {st.session_state.get('website_title', 'Website')}")
        if web_prompt := st.chat_input("Ask about the website...", key="web_chat"):
            st.session_state.website_messages.append({"role": "user", "content": web_prompt})
            
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
                                website_content = website_content[:max_content_length] + "..."
                            
                            llm = get_llm(model_name, temperature, max_tokens)
                            if llm is None:
                                error_msg = "⚠️ Website mode requires a Groq model and valid GROQ_API_KEY. Please configure in Settings or .env file."
                                st.error(error_msg)
                                st.session_state.website_messages.append({"role": "assistant", "content": error_msg})
                                logger.log_event("WEBSITE_ANSWER_SKIPPED", {"model_selected": model_name})
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
                            
                            prompt_template = ChatPromptTemplate.from_template(answer_prompt)
                            formatted_prompt = prompt_template.format(
                                website_title=website_title,
                                website_url=website_url,
                                website_content=website_content,
                                question=web_prompt,
                            )
                            
                            # Get answer using improved extraction
                            answer = extract_llm_response(llm, formatted_prompt)
                            
                            st.markdown(answer)
                            st.session_state.website_messages.append({"role": "assistant", "content": answer})
                            logger.log_chat_interaction("WEBSITE", len(web_prompt), len(answer))
                            
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            logger.log_error("WEBSITE_CHAT_ERROR", str(e), {"question": web_prompt[:100]})
                            st.error(error_msg)
                            st.session_state.website_messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
    else:
        st.info("👆 Enter a website URL and click 'Scrape Website' to start chatting!")


# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("💡 Tip: Use both modes simultaneously!")
st.sidebar.caption("📄 Left: PDF analysis only")
st.sidebar.caption("🌐 Right: Website content only")