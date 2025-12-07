"""
Chroma DB cleanup utility - can be imported into app.py
"""
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def safe_cleanup_chroma(logger=None):
    """
    Clean up any corrupted Chroma DB directories on startup.
    
    Args:
        logger: Optional logger instance to log events
        
    Returns:
        list: Directories that were cleaned up
    """
    directories_to_check = ["chroma_db", "chroma"]
    cleaned_directories = []
    
    for directory in directories_to_check:
        if os.path.exists(directory):
            try:
                # Try to verify if it's a valid Chroma DB
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
                temp_db = Chroma(persist_directory=directory, embedding_function=embeddings)
                temp_db.get()  # Test if DB is accessible
                
                if logger:
                    logger.log_event("CHROMA_CHECK", {"directory": directory, "status": "valid"})
                    
            except Exception as e:
                # DB is corrupted, remove it
                if logger:
                    logger.log_event("CHROMA_CLEANUP", {
                        "directory": directory,
                        "reason": str(e)
                    })
                
                try:
                    shutil.rmtree(directory, ignore_errors=True)
                    cleaned_directories.append(directory)
                    print(f"✅ Cleaned up corrupted database: {directory}")
                    
                except Exception as cleanup_error:
                    if logger:
                        logger.log_error("CHROMA_CLEANUP_FAILED", str(cleanup_error), {
                            "directory": directory
                        })
                    print(f"❌ Failed to clean up {directory}: {cleanup_error}")
    
    return cleaned_directories


def force_cleanup_chroma():
    """
    Force delete all Chroma DB directories without checking.
    Use this if you want a fresh start.
    """
    directories = ["chroma_db", "chroma"]
    removed = []
    
    for directory in directories:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(directory)
                print(f"🗑️  Removed: {directory}")
            except Exception as e:
                print(f"❌ Failed to remove {directory}: {e}")
    
    return removed


if __name__ == "__main__":
    print("🧹 Running Chroma DB cleanup...")
    cleaned = safe_cleanup_chroma()
    
    if cleaned:
        print(f"\n✅ Cleaned up {len(cleaned)} corrupted database(s): {', '.join(cleaned)}")
    else:
        print("\n✅ No corrupted databases found")
