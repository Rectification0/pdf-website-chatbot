"""
Simple script to initialize the log files without running the full app
"""

import json
import os
from datetime import datetime
import socket


def is_local_deployment():
    """Detect if running locally or on web"""
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


def initialize_log_file(log_file, deployment_type):
    """Create log file if it doesn't exist"""
    if not os.path.exists(log_file):
        initial_data = {
            "deployment_type": deployment_type,
            "created_at": datetime.now().isoformat(),
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "deployment": deployment_type,
                    "event_type": "LOG_INITIALIZED",
                    "status": "success",
                    "details": {"message": "Log file created"},
                }
            ],
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)
        print(f"✅ Created: {log_file}")
        return True
    else:
        print(f"ℹ️  Already exists: {log_file}")
        return False


if __name__ == "__main__":
    is_local = is_local_deployment()

    if is_local:
        log_file = "chatbot_local_logs.json"
        deployment_type = "LOCAL"
    else:
        log_file = "chatbot_web_logs.json"
        deployment_type = "WEB"

    print(f"🖥️  Deployment type: {deployment_type}")
    initialize_log_file(log_file, deployment_type)
    print(f"\n📊 Log file location: {os.path.abspath(log_file)}")
