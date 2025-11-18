"""
ElevenLabs Text-to-Speech Service
Handles TTS initialization and audio generation
"""

import os

# Default voice settings
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice
DEFAULT_VOICE_MODEL = "eleven_turbo_v2_5"

# Global state
_elevenlabs_client = None
_elevenlabs_available = False


def initialize_elevenlabs():
    """Initialize ElevenLabs TTS client"""
    global _elevenlabs_client, _elevenlabs_available
    
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_api_key:
            _elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
            _elevenlabs_available = True
            print("ElevenLabs TTS initialized")
            return True
        else:
            _elevenlabs_available = False
            _elevenlabs_client = None
            print("Warning: ELEVENLABS_API_KEY not found. Voice output disabled.")
            return False
    except Exception as e:
        _elevenlabs_available = False
        _elevenlabs_client = None
        print(f"Warning: Could not initialize ElevenLabs: {e}")
        return False


def is_available():
    """Check if ElevenLabs TTS is available"""
    return _elevenlabs_available


def text_to_speech(text, voice_id=DEFAULT_VOICE_ID, model_id=DEFAULT_VOICE_MODEL):
    """
    Convert text to speech using ElevenLabs
    
    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID (default: Adam voice)
        model_id: ElevenLabs model ID (default: eleven_turbo_v2_5)
    
    Returns:
        bytes: Audio data as MP3 bytes, or None if unavailable
    """
    if not _elevenlabs_available or not _elevenlabs_client:
        return None
    
    try:
        from elevenlabs import VoiceSettings
        
        # Clean text for TTS (remove markdown and sources section)
        text_for_tts = text.split("─" * 60)[0].strip()
        text_for_tts = text_for_tts.replace("*", "").replace("#", "")
        
        # Limit text to approximately 2 minutes of speech
        # Average speaking rate: ~150 words/min = ~300 words for 2 min
        # Average word length: ~5 chars, so ~2000 chars for 2 min (with buffer)
        MAX_CHARS_FOR_2_MIN = 2000
        
        if len(text_for_tts) > MAX_CHARS_FOR_2_MIN:
            # Truncate at word boundary to avoid cutting mid-word
            truncated = text_for_tts[:MAX_CHARS_FOR_2_MIN]
            # Find last space before the limit
            last_space = truncated.rfind(' ')
            if last_space > MAX_CHARS_FOR_2_MIN * 0.8:  # Only use if we're not losing too much
                text_for_tts = truncated[:last_space] + "..."
            else:
                text_for_tts = truncated + "..."
            print(f"Warning: Text truncated to ~2 minutes of audio (from {len(text)} to {len(text_for_tts)} chars)")
        
        # Generate speech using ElevenLabs
        audio_stream = _elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text_for_tts,
            voice_settings=VoiceSettings(
                stability=0.55,
                similarity_boost=0.75,
                style=0.15,
                use_speaker_boost=True
            )
        )
        
        # Convert generator to bytes
        audio_bytes = b''.join(audio_stream)
        return audio_bytes
        
    except Exception as e:
        print(f"Warning: ElevenLabs TTS error: {e}")
        return None


def get_client():
    """Get the ElevenLabs client instance (for advanced usage)"""
    return _elevenlabs_client

