import json
import os
import socket
from datetime import datetime
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            if os.getenv("STREAMLIT_SHARING_MODE") or os.getenv(
                "STREAMLIT_SERVER_HEADLESS"
            ):
                return False
            hostname = socket.gethostname()
            if "streamlit" in hostname.lower() or "cloud" in hostname.lower():
                return False
            return True
        except Exception:
            return True

    def _initialize_log_file(self):
        """Create log file if missing"""
        try:
            if not os.path.exists(self.log_file):
                initial_data = {
                    "deployment_type": self.deployment_type,
                    "created_at": datetime.now().isoformat(),
                    "logs": [],
                }
                with open(self.log_file, "w", encoding="utf-8") as f:
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
                "details": details,
            }
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {
                    "deployment_type": self.deployment_type,
                    "created_at": datetime.now().isoformat(),
                    "logs": [],
                }
            data["logs"].append(log_entry)
            with open(self.log_file, "w", encoding="utf-8") as f:
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
            "context": context or {},
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
        details = {
            "mode": mode,
            "question_length": question_length,
            "answer_length": answer_length,
        }
        return self.log_event("CHAT_INTERACTION", details, status="success")

    def get_log_summary(self):
        """Return aggregated log summary"""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_logs = len(data.get("logs", []))
            errors = sum(
                1 for log in data.get("logs", []) if log.get("status") == "error"
            )
            return {
                "total_events": total_logs,
                "total_errors": errors,
                "deployment": data.get("deployment_type", "UNKNOWN"),
                "created_at": data.get("created_at", "Unknown"),
            }
        except Exception:
            return None


# Initialize logger for use within backend functions if needed
# Note: Typically, we'd pass logger as a dependency, but for this refactor to be exact
# we can instantiate it or rely on the one passed from app. However,
# helper functions like `scrape_website` use `logger` globally in the original code.
# So we need a global logger instance here.
logger = ChatbotLogger()


# --- Helper Functions ---
def validate_url(url: str) -> dict:
    """Validate URL and check for problematic domains"""
    blocked_patterns = [
        "bank",
        "paypal",
        "stripe",
        "payment",
        "login",
        "signin",
        "auth",
        "account",
        "admin",
        "dashboard",
        "portal",
        "medical",
        "health",
        "patient",
        "gov",
        "military",
        "defense",
        "private",
        "internal",
        "intranet",
    ]
    suspicious_tlds = [".onion", ".i2p"]
    try:
        parsed = urlparse(url.lower())
        if not parsed.scheme:
            return {"valid": False, "error": "Invalid URL: Missing http:// or https://"}
        if parsed.scheme not in ["http", "https"]:
            return {
                "valid": False,
                "error": "Only HTTP and HTTPS protocols are allowed",
            }
        if any(
            x in parsed.netloc
            for x in ["localhost", "127.0.0.1", "0.0.0.0", "192.168.", "10."]
        ):
            return {
                "valid": False,
                "error": "Cannot scrape local or private network addresses",
            }
        for pattern in blocked_patterns:
            if pattern in parsed.netloc or pattern in parsed.path:
                return {
                    "valid": False,
                    "error": f"This appears to be a sensitive website ({pattern}). Scraping is not allowed for security and privacy reasons.",
                }
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
                return {
                    "allowed": False,
                    "warning": "This website's robots.txt discourages scraping. Please respect their wishes.",
                }
        return {"allowed": True}
    except Exception:
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
            logger.log_error(
                "ROBOTS_TXT_DISALLOW", robots_check["warning"], {"url": url}
            )
            return {"success": False, "error": robots_check["warning"]}
        logger.log_event("WEBSITE_SCRAPE_START", {"url": url})
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "iframe"]
        ):
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
        return {
            "success": True,
            "title": title_text,
            "content": text,
            "url": url,
            "warning": robots_check.get("warning"),
        }
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch URL: {str(e)}"
        logger.log_error("WEBSITE_SCRAPE_ERROR", error_msg, {"url": url})
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error scraping website: {str(e)}"
        logger.log_error("WEBSITE_SCRAPE_ERROR", error_msg, {"url": url})
        return {"success": False, "error": error_msg}


# --- Cache the Embedding Model (CRITICAL FIX) ---
@st.cache_resource
def get_shared_embeddings():
    """
    Loads the embedding model ONLY ONCE and shares it across all users.
    This prevents the app from crashing due to running out of memory.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings():
    """Return HuggingFace embeddings instance"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
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
            streaming=False,
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
        if hasattr(response, "content"):
            # AIMessage with content attribute
            return str(response.content)
        elif isinstance(response, str):
            # Direct string response
            return response
        elif isinstance(response, dict) and "text" in response:
            # Dict with text key
            return response["text"]
        elif isinstance(response, dict) and "content" in response:
            # Dict with content key
            return response["content"]
        else:
            # Fallback: convert to string
            response_str = str(response)
            # Log the response type for debugging
            logger.log_event(
                "LLM_RESPONSE_TYPE",
                {"type": str(type(response)), "preview": response_str[:100]},
            )
            return response_str

    except Exception as e:
        logger.log_error("LLM_INVOKE_ERROR", str(e))
        return f"Error calling LLM: {str(e)}"
