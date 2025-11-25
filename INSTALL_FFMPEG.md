# Install FFmpeg for ElevenLabs TTS

The voice coach uses ElevenLabs for high-quality text-to-speech, which requires FFmpeg for audio processing.

## Windows Installation

### Option 1: Using Chocolatey (Recommended)
```bash
choco install ffmpeg
```

### Option 2: Manual Installation
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract the ZIP file
3. Add the `bin` folder to your PATH:
   - Open "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the path to FFmpeg's `bin` folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs
4. Restart your terminal

### Option 3: Using Scoop
```bash
scoop install ffmpeg
```

## Verify Installation

```bash
ffmpeg -version
```

You should see FFmpeg version information.

## After Installation

1. Restart the voice coach server:
```bash
cd tennis-ai-main
python voice_coach_server.py
```

2. The backend will now use ElevenLabs TTS for high-quality voice responses

## Current Status

- ✅ **RAG (Pinecone Vector Database)** - Working
- ✅ **Claude 4 Sonnet AI** - Working  
- ✅ **Voice Recognition** - Working
- ✅ **Browser TTS** - Working (fallback)
- ⚠️ **ElevenLabs TTS** - Requires FFmpeg

Without FFmpeg, the system uses browser TTS which still works but with lower quality audio.

