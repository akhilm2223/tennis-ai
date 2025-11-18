# Voice Coaching Setup Guide

The Mental Coaching feature requires a separate backend server on port 5002.

## Required Dependencies

Install these packages to enable voice coaching:

```bash
pip install SpeechRecognition pydub python-dotenv pinecone-client anthropic google-generativeai elevenlabs
```

## Environment Variables

Create a `.env` file in the project root:

```env
# AI API Keys (choose one)
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Voice TTS
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# Vector Database (for RAG)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_environment
```

## Starting the Voice Coach Server

```bash
python voice_coach_server.py
```

The server will run on `http://localhost:5002`

## Features

- **Voice Input**: Record your questions using the microphone
- **AI Coaching**: RAG-based mental coaching using tennis psychology books
- **Voice Output**: Text-to-speech responses using ElevenLabs

## Architecture

1. **Frontend** (React): Records audio and sends to backend
2. **voice_coach_server.py** (Port 5002): Handles voice API requests
3. **RAG_MentalCoach/**: Contains coaching knowledge base and AI logic
4. **tts_service.py**: Text-to-speech conversion

## Current Status

✅ Video Analysis working (port 5001)
🔧 Voice Coaching requires setup (port 5002)

**Next Steps:**
- Set up API keys
- Install dependencies
- Start voice_coach_server.py
- Test voice/text coaching features

