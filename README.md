# 🤖 AI-Powered PDF + Web Scraping Chatbot

An intelligent chatbot that combines PDF content analysis with real-time web scraping to answer your questions comprehensively.

## Features

- 📄 Upload and process PDF documents
- 🔍 Real-time web scraping for up-to-date information
- ⚡ Lightning-fast responses using Groq's LLMs
- 🌐 Source citations for transparency
- 💬 Interactive chat interface

## Setup Instructions

### 1. Install Dependencies

Run the setup script:
```bash
setup.bat
```

Or manually install:
```bash
pip install -r requirements.txt
```

### 2. Get Your Groq API Key

1. Visit https://console.groq.com/keys
2. Sign up for a free account
3. Create a new API key

### 3. Configure API Key

Edit `.streamlit/secrets.toml` and add your key:
```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

Or set as environment variable:
```bash
set GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## How to Use

1. Upload a PDF file
2. Click "Process PDF" to index the document
3. Ask questions about the PDF
4. Get answers enhanced with web-scraped information

## Troubleshooting

**Import Errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.9 or higher

**API Key Errors:**
- Verify your Groq API key is correct
- Check that `.streamlit/secrets.toml` exists and has the correct format

**PDF Processing Errors:**
- Ensure the PDF is not corrupted
- Try a different PDF file
- Check that the PDF contains extractable text (not just images)

## Models Available

- llama-3.1-70b-versatile (most capable)
- llama-3.1-8b-instant (fastest)
- mixtral-8x7b-32768 (large context)
- gemma2-9b-it (balanced)
