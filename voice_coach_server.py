"""
Voice Coach Server - Mental Advice Service
Simple Flask server for voice-based mental coaching
"""

import os
import dotenv
dotenv.load_dotenv()

from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Import coach chatbot
try:
    from RAG_MentalCoach.coach.coach import MentalCoachChatbot
    coach_chatbot = MentalCoachChatbot()
    COACH_AVAILABLE = True
    print("Mental Coach Chatbot initialized")
except Exception as e:
    print(f"Warning: Could not initialize Mental Coach Chatbot: {e}")
    COACH_AVAILABLE = False
    coach_chatbot = None

# Initialize TTS service
try:
    from tts_service import initialize_elevenlabs, is_available as tts_is_available, text_to_speech
    initialize_elevenlabs()
    ELEVENLABS_AVAILABLE = tts_is_available()
except Exception as e:
    ELEVENLABS_AVAILABLE = False
    print(f"Warning: Could not import TTS service: {e}")

# Initialize Speech Recognition for STT
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    SPEECH_RECOGNITION_AVAILABLE = True
    print("Speech Recognition initialized")
except ImportError as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    print(f"Warning: Could not import Speech Recognition libraries: {e}")
    print("   Install with: pip install SpeechRecognition pydub")
except Exception as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    print(f"Warning: Could not initialize Speech Recognition: {e}")


def sanitize_header_value(value):
    """
    Sanitize a value for use in HTTP headers.
    HTTP headers must contain only latin-1 characters (code points 0-255).
    Removes characters outside the latin-1 range.
    """
    if not value:
        return ''
    try:
        # Try encoding as latin-1, which will fail if there are code points > 255
        value.encode('latin-1')
        return value
    except UnicodeEncodeError:
        # Remove characters outside latin-1 range (code points > 255)
        # latin-1 maps directly to Unicode code points 0-255
        sanitized = ''.join(char for char in value if ord(char) < 256)
        return sanitized


@app.route('/', methods=['GET'])
@app.route('/index.html', methods=['GET'])
def serve_index():
    """Serve the index.html file"""
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    return send_file(index_path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'coach_available': COACH_AVAILABLE,
        'tts_available': ELEVENLABS_AVAILABLE,
        'stt_available': SPEECH_RECOGNITION_AVAILABLE
    })


@app.route('/api/coach/voice-chat', methods=['POST'])
def coach_voice_chat():
    """Voice chat endpoint: accepts audio, transcribes, gets RAG response, returns audio"""
    if not COACH_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Coach chatbot not available. Check API keys and configuration.'
        }), 503
    
    if not SPEECH_RECOGNITION_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Speech recognition not available. Install SpeechRecognition library.'
        }), 503
    
    if not ELEVENLABS_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'ElevenLabs TTS not available. Check ELEVENLABS_API_KEY.'
        }), 503
    
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided. Use form field "audio".'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty audio file'
            }), 400
        
        print(f"Received audio file: {audio_file.filename}")
        
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_path = temp_audio.name
            audio_file.save(audio_path)
        
        wav_path = None
        try:
            # Convert audio to WAV format if needed (for speech recognition)
            try:
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.replace('.wav', '_converted.wav')
                audio.export(wav_path, format="wav")
                del audio  # Release file handle
            except FileNotFoundError as e:
                if 'ffprobe' in str(e) or 'ffmpeg' in str(e):
                    return jsonify({
                        'success': False,
                        'error': 'FFmpeg not found. Please install FFmpeg:\n'
                                '  macOS: brew install ffmpeg\n'
                                '  Linux: sudo apt-get install ffmpeg\n'
                                '  Windows: Download from https://ffmpeg.org/download.html'
                    }), 500
                raise
            
            # Transcribe audio using Google Speech Recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            
            # Transcribe
            try:
                query = recognizer.recognize_google(audio_data)
                print(f"Transcribed: {query}")
            except sr.UnknownValueError:
                return jsonify({
                    'success': False,
                    'error': 'Could not understand audio. Please speak more clearly.'
                }), 400
            except sr.RequestError as e:
                return jsonify({
                    'success': False,
                    'error': f'Speech recognition service error: {str(e)}'
                }), 500
            
            # Get session ID from request (or use default)
            session_id = request.form.get('session_id', 'default')
            if not session_id or session_id.strip() == '':
                session_id = 'default'
            
            # Get RAG response with conversation memory
            context_items = coach_chatbot.search_pinecone(query)
            response_text = coach_chatbot.generate_response(query, context_items, session_id=session_id)
            
            if not response_text:
                return jsonify({
                    'success': False,
                    'error': 'Could not generate response'
                }), 500
            
            print(f"Generated response: {response_text[:100]}...")
            
            # Extract sources from context_items
            sources_list = []
            for idx, item in enumerate(context_items, 1):
                source_info = {}
                
                # Get title
                title = (
                    item.get("book_title")
                    or item.get("article_title")
                    or item.get("podcast_title")
                    or item.get("interview_title")
                    or item.get("youtube_title")
                    or item.get("title")
                    or "Unknown Source"
                )
                source_info['title'] = title
                
                # Get author/creator
                author = (
                    item.get("book_author")
                    or item.get("article_author")
                    or item.get("author")
                    or item.get("host")
                    or item.get("channel_name")
                    or None
                )
                if author:
                    source_info['author'] = author
                
                # Get source type
                content_type = item.get('content_type', 'Unknown')
                source_info['type'] = content_type
                
                # Get page number if available
                if 'page_number' in item:
                    source_info['page'] = item['page_number']
                
                # Get URL if available
                url = item.get("url") or item.get("source_url")
                if url:
                    source_info['url'] = url
                
                # Get relevance score
                if 'score' in item:
                    source_info['score'] = round(item['score'], 2)
                
                sources_list.append(source_info)
            
            # Format sources for header (JSON string, limited length)
            import json
            sources_json = json.dumps(sources_list[:5])  # Limit to first 5 sources
            sources_header = sources_json[:2000] if len(sources_json) <= 2000 else sources_json[:1997] + "..."
            
            # Log sources used in the response
            print(f"Sources used in response ({len(sources_list)} total):")
            for idx, source in enumerate(sources_list, 1):
                source_str = f"  {idx}. [{source.get('type', 'Unknown')}] {source.get('title', 'Unknown Source')}"
                if source.get('author'):
                    source_str += f" by {source['author']}"
                if source.get('page'):
                    source_str += f" (Page {source['page']})"
                if source.get('score'):
                    source_str += f" [Relevance: {source['score']}]"
                if source.get('url'):
                    source_str += f" | {source['url']}"
                print(source_str)
            
            # Generate speech using ElevenLabs TTS service
            # For TTS, use only the response text without sources section (sources are separated in the text)
            audio_bytes = text_to_speech(response_text)
            
            if audio_bytes is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not generate audio response'
                }), 500
            
            # Clean up temp files (use try-except to handle file locking issues on Windows)
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {audio_path}: {e}")
            
            try:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {wav_path}: {e}")
            
            # Return audio file
            # Sanitize header values (remove newlines and non-ASCII characters)
            sanitized_query = query.replace('\n', ' ').replace('\r', ' ') if query else ''
            sanitized_query = sanitize_header_value(sanitized_query)
            
            # Include full response text (or at least more of it)
            sanitized_response = response_text.replace('\n', ' ').replace('\r', ' ') if response_text else ''
            # Truncate if too long for header (HTTP headers have size limits)
            if len(sanitized_response) > 8000:
                sanitized_response = sanitized_response[:7997] + "..."
            sanitized_response = sanitize_header_value(sanitized_response)
            
            # Sanitize sources header
            sanitized_sources = sanitize_header_value(sources_header)
            
            return Response(
                audio_bytes,
                mimetype='audio/mpeg',
                headers={
                    'Content-Disposition': 'attachment; filename=coach_response.mp3',
                    'X-Transcribed-Text': sanitized_query,
                    'X-Response-Text': sanitized_response,  # Full response text
                    'X-Sources': sanitized_sources  # JSON array of sources
                }
            )
            
        except Exception as e:
            # Clean up temp files on error (use try-except to handle file locking issues)
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temp file {audio_path}: {cleanup_error}")
            
            try:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temp file {wav_path}: {cleanup_error}")
            raise e
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in voice chat: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/coach/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a session (memory not available)"""
    return jsonify({
        'success': True,
        'message': 'Memory functionality is not enabled. No conversation history to clear.'
    })


@app.route('/api/coach/chat-with-audio', methods=['POST'])
def coach_chat_with_audio():
    """Text-based chat that returns both text and ElevenLabs audio"""
    if not COACH_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Coach chatbot not available. Check API keys and configuration.'
        }), 503
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400
    
    try:
        # Get session ID from request (or use default)
        session_id = data.get('session_id', 'default')
        if not session_id or session_id.strip() == '':
            session_id = 'default'
        
        # Search for relevant content
        context_items = coach_chatbot.search_pinecone(query)
        
        # Generate response with conversation memory
        response_text = coach_chatbot.generate_response(query, context_items, session_id=session_id)
        
        if not response_text:
            return jsonify({
                'success': False,
                'error': 'Could not generate response'
            }), 500
        
        # Generate ElevenLabs audio if available
        print(f"ELEVENLABS_AVAILABLE: {ELEVENLABS_AVAILABLE}")
        if ELEVENLABS_AVAILABLE:
            try:
                print("Generating ElevenLabs audio...")
                audio_bytes = text_to_speech(response_text)
                print(f"Audio bytes generated: {len(audio_bytes) if audio_bytes else 0}")
                if audio_bytes:
                    print("Returning audio response")
                    # Return audio with text in headers
                    return Response(
                        audio_bytes,
                        mimetype='audio/mpeg',
                        headers={
                            'X-Response-Text': sanitize_header_value(response_text),
                            'X-Query': sanitize_header_value(query)
                        }
                    )
                else:
                    print("ElevenLabs returned None - falling back to JSON")
            except Exception as e:
                print(f"ElevenLabs TTS error: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to return text-only response
        else:
            print("ElevenLabs not available - returning JSON")
        
        # Fallback: return text-only response
        return jsonify({
            'success': True,
            'response': response_text,
            'audio_available': False
        })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in coach chat with audio: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/coach/chat', methods=['POST'])
def coach_chat():
    """Text-based chat with the mental coach chatbot (for testing)"""
    if not COACH_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Coach chatbot not available. Check API keys and configuration.'
        }), 503
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400
    
    try:
        # Get session ID from request (or use default)
        session_id = data.get('session_id', 'default')
        if not session_id or session_id.strip() == '':
            session_id = 'default'
        
        # Search for relevant content
        context_items = coach_chatbot.search_pinecone(query)
        
        # Generate response with conversation memory
        response = coach_chatbot.generate_response(query, context_items, session_id=session_id)
        
        # Extract and log sources used in the response
        if context_items:
            print(f"Sources used in response ({len(context_items)} total):")
            for idx, item in enumerate(context_items, 1):
                # Get title
                title = (
                    item.get("book_title")
                    or item.get("article_title")
                    or item.get("podcast_title")
                    or item.get("interview_title")
                    or item.get("youtube_title")
                    or item.get("title")
                    or "Unknown Source"
                )
                
                # Get author/creator
                author = (
                    item.get("book_author")
                    or item.get("article_author")
                    or item.get("author")
                    or item.get("host")
                    or item.get("channel_name")
                    or None
                )
                
                # Get source type
                content_type = item.get('content_type', 'Unknown')
                
                source_str = f"  {idx}. [{content_type}] {title}"
                if author:
                    source_str += f" by {author}"
                if item.get('page_number'):
                    source_str += f" (Page {item['page_number']})"
                if item.get('score'):
                    source_str += f" [Relevance: {round(item['score'], 2)}]"
                url = item.get("url") or item.get("source_url")
                if url:
                    source_str += f" | {url}"
                print(source_str)
        else:
            print("No sources found for this query")
        
        if response:
            return jsonify({
                'success': True,
                'response': response,
                'sources_count': len(context_items) if context_items else 0
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate response'
            }), 500
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in coach chat: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Voice Coach Server Starting...")
    print("Server running on http://localhost:5002")
    print("Web interface: http://localhost:5002/")
    print("Voice chat endpoint: POST /api/coach/voice-chat")
    print("Text chat endpoint: POST /api/coach/chat")
    app.run(host='0.0.0.0', port=5002, debug=True)

