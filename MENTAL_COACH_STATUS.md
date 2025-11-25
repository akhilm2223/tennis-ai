# Mental Coach Status

## Current Status

The Mental Coach server (`voice_coach_server.py`) is running on port 5002, but the full RAG (Retrieval Augmented Generation) system is currently **not fully functional** due to venv dependency issues.

##Issues Found

1. **Broken venv packages**: `certifi`, `python-dateutil`, and `colorama` have corrupted installations in venv
2. **Pinecone connection**: The Pinecone vector database index "mental-coaching" needs to be created/configured

## What Works

- ✅ Flask server runs on port 5002
- ✅ API endpoints are accessible
- ✅ Server gracefully handles missing RAG components
- ✅ Basic mental coaching can work without RAG (using Claude directly)

## What's Affected

- ⚠️ Knowledge retrieval from Pinecone vector database won't work
- ⚠️ RAG-enhanced responses won't include tennis psychology book references

## Quick Fix Options

### Option 1: Rebuild venv (Recommended)
```bash
cd tennis-ai-main
rm -rf venv
python -m venv venv
source venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

### Option 2: Fix broken packages manually
```bash
cd tennis-ai-main
source venv/Scripts/activate
rm -rf venv/Lib/site-packages/certifi*
rm -rf venv/Lib/site-packages/python_dateutil*
pip install certifi python-dateutil colorama
pip install "pinecone[grpc]==5.3.1"
```

### Option 3: Use without RAG
The system will work without Pinecone - Claude will provide coaching based on its training alone.

## Pinecone Setup (for full RAG)

If you want the full knowledge-augmented system:

1. Create a Pinecone index named "mental-coaching" in your Pinecone dashboard
2. Populate it with tennis psychology content
3. Ensure `.env` has valid `PINECONE_API_KEY`

## Testing

Test the server:
```bash
curl http://localhost:5002/api/health
```

Expected response:
```json
{
  "status": "online",
  "coach_available": true/false,
  "tts_available": true/false
}
```

