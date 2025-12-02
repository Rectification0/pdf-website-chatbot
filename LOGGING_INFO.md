# 📊 Chatbot Logging System

## Overview

The chatbot automatically logs all events, errors, and interactions with real-time timestamps.

## Log Files

### 1. `chatbot_local_logs.json`
- Created when running the app **locally** on your machine
- Tracks all local development and testing activities
- Located in the project root directory

### 2. `chatbot_web_logs.json`
- Created when the app is deployed on **Streamlit Cloud** or other web hosting
- Tracks all production/public usage
- Separate from local logs for security and organization

## What Gets Logged

### Events Tracked:
- ✅ **App Start** - When the application launches
- 📄 **PDF Processing** - File uploads, chunk counts, success/failure
- 🌐 **Website Scraping** - URLs scraped, content length, success/failure
- 💬 **Chat Interactions** - Question/answer lengths, mode (PDF/Website)
- ❌ **Errors** - All errors with context and details
- 🗄️ **Vector Database** - Creation and operations

### Log Entry Format:
```json
{
  "timestamp": "2024-12-01T14:30:45.123456",
  "deployment": "LOCAL" or "WEB",
  "event_type": "PDF_PROCESSING",
  "status": "success" or "error",
  "details": {
    "filename": "document.pdf",
    "chunks_count": 25
  }
}
```

## Viewing Logs

### In the App:
- Check the **sidebar** for a live summary:
  - Deployment type (LOCAL/WEB)
  - Total events logged
  - Total errors encountered
  - Start date

### In the File:
- Open `chatbot_local_logs.json` or `chatbot_web_logs.json`
- View complete history with timestamps
- Analyze patterns and troubleshoot issues

## Log File Structure

```json
{
  "deployment_type": "LOCAL",
  "created_at": "2024-12-01T10:00:00.000000",
  "logs": [
    {
      "timestamp": "2024-12-01T10:00:01.123456",
      "deployment": "LOCAL",
      "event_type": "APP_START",
      "status": "success",
      "details": {
        "message": "Application started"
      }
    },
    ...
  ]
}
```

## Privacy & Security

- **Local logs** stay on your machine
- **Web logs** are stored on the server (if deployed)
- No sensitive data (API keys, personal info) is logged
- Only operational metrics and error information

## Troubleshooting

### If logs aren't being created:
1. Check file permissions in the project directory
2. Ensure the app has write access
3. Check the sidebar for error counts

### If you want to clear logs:
- Simply delete the `.json` log files
- They will be recreated automatically on next app start

## Benefits

- 🔍 **Debug issues** - See exactly what went wrong and when
- 📈 **Track usage** - Monitor how the app is being used
- 🛡️ **Error monitoring** - Catch and fix problems quickly
- 📊 **Analytics** - Understand user behavior patterns
- 🕐 **Audit trail** - Complete history of all operations
