import streamlit as st
import tempfile
import os

# --- Page Config ---
st.set_page_config(page_title="AI Web-Scraping PDF Chatbot", layout="centered")
st.title("🤖 AI-Powered PDF + Web Scraping Chatbot")
st.caption("Ask questions about your PDF and get answers enhanced with real-time web search")

# Check imports and show helpful error messages
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import quote_plus
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Please run: pip install -r requirements.txt")
    st.stop()

# Model selection with descriptions
st.sidebar.header("⚙️ Settings")

model_options = {
    "llama-3.3-70b-versatile": "🚀 Llama 3.3 70B - Best overall (Recommended)",
    "llama-3.1-70b-versatile": "⚡ Llama 3.1 70B - Very capable",
    "llama-3.2-90b-text-preview": "🔥 Llama 3.2 90B - Most powerful",
    "mixtral-8x7b-32768": "📚 Mixtral 8x7B - Large context (32k tokens)",
    "llama-3.1-8b-instant": "💨 Llama 3.1 8B - Fastest",
}

model_name = st.sidebar.selectbox(
    "Choose AI Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0,
)

# Temperature control
temperature = st.sidebar.slider(
    "Temperature (creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Lower = more focused, Higher = more creative"
)

# Max tokens
max_tokens = st.sidebar.slider(
    "Max response length",
    min_value=512,
    max_value=4096,
    value=2048,
    step=256,
    help="Maximum length of AI responses"
)


def search_web(query: str, num_results: int = 2) -> list:
    """Search the web and scrape content from top results"""
    scraped_content = []
    try:
        # Use DuckDuckGo HTML search (more reliable than Google)
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract links from DuckDuckGo results
        links = []
        for result in soup.find_all("a", class_="result__a")[:num_results]:
            url = result.get("href")
            if url and url.startswith("http"):
                links.append(url)

        # Scrape content from each link
        for url in links:
            try:
                page_response = requests.get(url, headers=headers, timeout=8)
                page_soup = BeautifulSoup(page_response.text, "html.parser")

                # Remove unwanted elements
                for element in page_soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Extract text
                text = page_soup.get_text(separator=" ", strip=True)

                # Clean and limit text
                text = " ".join(text.split())[:1500]

                if len(text) > 100:  # Only add if we got meaningful content
                    scraped_content.append({"url": url, "content": text})
            except Exception:
                continue

    except Exception as e:
        st.warning(f"Web search encountered an issue: {str(e)}")

    return scraped_content


def get_embeddings():
    """Get embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )


def create_vectordb(documents):
    """Create a new vector database from documents"""
    try:
        embeddings = get_embeddings()
        vectordb = Chroma.from_documents(
            documents=documents, embedding=embeddings, collection_name="pdf_collection"
        )
        return vectordb
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        st.stop()


def get_llm(model: str, temp: float = 0.7, max_tok: int = 2048):
    """Get Groq LLM instance with custom parameters"""
    try:
        # Try multiple ways to get API key
        api_key = None

        # Method 1: Streamlit secrets
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]

        # Method 2: Environment variable
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key or api_key == "your-groq-api-key-here":
            st.error("⚠️ Please add your GROQ_API_KEY!")
            st.info(
                """
            **How to add your API key:**
            1. Get a free API key from: https://console.groq.com/keys
            2. Add it to `.streamlit/secrets.toml`:
               ```
               GROQ_API_KEY = "your-actual-key-here"
               ```
            3. Or set environment variable: `set GROQ_API_KEY=your-key`
            """
            )
            st.stop()

        return ChatGroq(
            model=model, 
            groq_api_key=api_key, 
            temperature=temp, 
            max_tokens=max_tok,
            streaming=False
        )
    except Exception as e:
        st.error(f"Error initializing Groq: {e}")
        st.stop()


# --- PDF Upload & Processing ---
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    if st.button("🚀 Process PDF", type="primary"):
        with st.spinner("Processing PDF..."):
            try:
                # Save uploaded file temporarily
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_pdf.write(uploaded_file.read())
                temp_pdf.close()

                # Load and split PDF
                loader = PyPDFLoader(temp_pdf.name)
                docs = loader.load()

                if not docs:
                    st.error("Could not extract text from PDF. Please try another file.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Create new vector database with documents
                vectordb = create_vectordb(splits)

                st.session_state.vectordb = vectordb
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.success(
                    f"✅ PDF processed! {len(splits)} chunks indexed from '{uploaded_file.name}'"
                )

                # Clean up temp file
                try:
                    os.unlink(temp_pdf.name)
                except:
                    pass

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.info("Please make sure the PDF is not corrupted and try again.")


# --- Chat Interface ---
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.get("pdf_processed"):
    st.caption(f"💬 Chatting about: {st.session_state.get('pdf_name', 'PDF')}")

    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Step 1: Get PDF context
                with st.spinner("📄 Analyzing PDF..."):
                    vectordb = st.session_state.vectordb
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    pdf_docs = retriever.invoke(prompt)
                    pdf_context = "\n\n".join([doc.page_content for doc in pdf_docs])

                # Step 2: Let LLM decide if web search is needed
                llm = get_llm(model_name, temperature, max_tokens)

                decision_prompt = """You are an advanced AI assistant analyzing a question about a PDF document.

PDF Context:
{pdf_context}

Question: {question}

Task: Analyze whether you can provide a complete, accurate answer using ONLY the PDF content, or if you need additional web information.

Consider:
- Is the question asking for current/recent information not in the PDF?
- Does it require external context, definitions, or comparisons?
- Would web sources significantly enhance the answer quality?

Respond with ONLY ONE of these:
- "SUFFICIENT" - PDF content is enough for a complete answer
- "NEED_WEB" - Additional web information would be valuable

Your decision:"""

                decision_template = ChatPromptTemplate.from_template(decision_prompt)
                decision_formatted = decision_template.format(
                    pdf_context=pdf_context if pdf_context else "No relevant content found",
                    question=prompt,
                )

                decision_response = llm.invoke(decision_formatted)
                needs_web = "NEED_WEB" in decision_response.content.upper()

                # Step 3: Search web only if needed
                web_results = []
                web_context = ""

                if needs_web:
                    with st.spinner("🌐 Searching web for additional context..."):
                        web_results = search_web(prompt, num_results=2)
                        web_context = "\n\n".join(
                            [f"Source: {result['url']}\n{result['content']}" for result in web_results]
                        )

                # Step 4: Generate final answer
                with st.spinner("💭 Generating answer..."):
                    if needs_web and web_context:
                        answer_prompt = """You are an advanced AI assistant with expertise in analyzing documents and synthesizing information from multiple sources.

PDF Context:
{pdf_context}

Web Search Results:
{web_context}

Instructions:
- Provide a comprehensive, well-reasoned answer using both sources
- Synthesize information intelligently, don't just concatenate
- Cite web sources when using external information (e.g., "According to [source]...")
- If information conflicts, prioritize the PDF content and explain any discrepancies
- Structure your answer with clear paragraphs
- Be conversational, insightful, and helpful
- Use examples when relevant

Question: {question}

Provide a detailed, high-quality answer:"""
                    else:
                        answer_prompt = """You are an advanced AI assistant with expertise in document analysis and comprehension.

PDF Context:
{pdf_context}

Instructions:
- Provide a thorough answer using the information from the PDF
- Apply your reasoning and analytical skills to provide insights
- Structure your answer clearly with proper paragraphs
- If relevant, make connections between different parts of the document
- Be accurate, clear, and helpful
- If the PDF doesn't contain enough information, acknowledge this honestly
- Use a conversational yet professional tone

Question: {question}

Provide a detailed, high-quality answer:"""

                    answer_template = ChatPromptTemplate.from_template(answer_prompt)
                    answer_formatted = answer_template.format(
                        pdf_context=pdf_context if pdf_context else "No relevant PDF content found",
                        web_context=web_context if web_context else "",
                        question=prompt,
                    )

                    response = llm.invoke(answer_formatted)
                    answer = response.content

                    st.markdown(answer)

                    # Show sources and reasoning
                    col1, col2 = st.columns(2)
                    with col1:
                        if needs_web:
                            st.caption("🌐 Used web search")
                        else:
                            st.caption("📄 PDF only")

                    if web_results:
                        with st.expander("🔗 Web Sources"):
                            for i, result in enumerate(web_results, 1):
                                st.write(f"{i}. [{result['url']}]({result['url']})")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.info("👆 Upload a PDF and click 'Process PDF' to begin chatting!")

    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
        1. **Upload a PDF** using the file uploader above
        2. **Click 'Process PDF'** to index the document
        3. **Ask questions** about the PDF content
        4. The AI will intelligently decide when to search the web for additional context
        
        **Features:**
        - Fast responses using Groq's LLMs
        - Intelligent web scraping only when needed
        - Source citations for transparency
        - Multiple model options
        """
        )
