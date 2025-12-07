# Chroma DB Cleanup Integration Guide

## Problem
Chroma vector databases can become corrupted, causing the app to crash on startup or when processing PDFs.

## Solution
I've created `chroma_cleanup.py` with two approaches:

### Option 1: Safe Cleanup (Recommended)
Only removes corrupted databases, keeps valid ones.

**Add to app.py after imports:**

```python
from chroma_cleanup import safe_cleanup_chroma

# After initializing logger, add:
cleaned = safe_cleanup_chroma(logger)
if cleaned:
    st.toast(f"🧹 Cleaned up corrupted databases: {', '.join(cleaned)}", icon="✅")
```

### Option 2: Force Cleanup
Always removes Chroma directories on startup (fresh start every time).

**Add to app.py after imports:**

```python
from chroma_cleanup import force_cleanup_chroma

# Run on startup:
force_cleanup_chroma()
```

### Option 3: Manual Cleanup Script
Run this command before starting the app:

```bash
python chroma_cleanup.py
```

## Integration Steps

1. **Import the cleanup function** (add after other imports):
```python
from chroma_cleanup import safe_cleanup_chroma
```

2. **Call it after logger initialization** (around line 180):
```python
# Initialize logger
logger = ChatbotLogger()
logger.log_event("APP_START", {"message": "Application started"})

# Clean up corrupted Chroma DBs
try:
    cleaned = safe_cleanup_chroma(logger)
    if cleaned:
        st.toast(f"🧹 Cleaned up: {', '.join(cleaned)}", icon="✅")
except Exception as e:
    logger.log_error("CHROMA_CLEANUP_ERROR", str(e))
```

3. **Add error handling to vector DB creation** (in `create_vectordb` function):
```python
def create_vectordb(documents):
    """Create a new vector database from documents"""
    try:
        logger.log_event("VECTORDB_CREATE_START", {"num_documents": len(documents)})
        
        embeddings = get_embeddings()
        
        # Clean up any existing corrupted DB before creating new one
        if os.path.exists("chroma_db"):
            try:
                shutil.rmtree("chroma_db", ignore_errors=True)
            except:
                pass
        
        vectordb = Chroma.from_documents(
            doc