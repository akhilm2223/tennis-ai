"""
Flask Backend Server for Tennis Analysis
Handles video uploads and processing with real-time updates
"""
# Fix PyTorch 2.6 weights_only issue - monkey patch ultralytics
import torch

# Monkey patch torch.load to use weights_only=False
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import cv2
import threading
import time
from werkzeug.utils import secure_filename
import base64
import json
import sys
import io

# Import the streaming version with callback for frame previews
from main_pose_streaming import main_with_callback


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Store processing status
processing_status = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_with_updates(video_path, output_path, session_id):
    """Process video with FULL analysis (same as main_pose.py)"""
    
    # Prepare JSON analysis file - use session_id in filename for easy lookup
    json_output_path = os.path.join(OUTPUT_FOLDER, f"analyzed_{session_id}_analysis.json")
    
    try:
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'starting',
            'message': 'Initializing FULL analysis (with rally tracking, statistics)...',
            'progress': 0
        })
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'processing',
            'message': f'Running complete analysis on {total_frames} frames...',
            'progress': 10,
            'total_frames': total_frames,
            'fps': fps
        })
        
        # DON'T capture stdout - let all messages print directly to console!
        print(f"\n{'='*60}", flush=True)
        print(f"🎾 STARTING FULL ANALYSIS", flush=True)
        print(f"📁 Video: {video_path}", flush=True)
        print(f"📊 Session: {session_id}", flush=True)
        print(f"💾 Output: {output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Progress callback for frame-by-frame updates WITH video frames AND logs
        def on_frame_processed(frame_num, total, frame_image, log_message=None):
            progress = int((frame_num / total) * 90) + 5  # 5-95%
            
            # Encode frame as JPEG for preview
            frame_preview = None
            if frame_image is not None:
                # Resize for faster transmission
                small_frame = cv2.resize(frame_image, (640, 360))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_preview = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame preview update
            socketio.emit('processing_update', {
                'session_id': session_id,
                'status': 'processing',
                'message': f'Processing frame {frame_num}/{total}...',
                'progress': progress,
                'frame_num': frame_num,
                'total_frames': total,
                'frame_preview': f'data:image/jpeg;base64,{frame_preview}' if frame_preview else None
            })
            
            # Send log message if provided
            if log_message:
                print(log_message, flush=True)
                socketio.emit('analysis_log', {
                    'session_id': session_id,
                    'log': log_message,
                    'timestamp': time.time()
                })
        
        try:
            print(f"🎬 Processing video with frame previews...", flush=True)
            
            # Process video with callback for frame previews
            result = main_with_callback(
                video_path=video_path,
                output_path=output_path,
                calibrate=True,
                use_pose=True,
                show_bbox=False,
                progress_callback=on_frame_processed
            )
            
            terminal_output = "Analysis complete"
            print(f"\n{'='*60}", flush=True)
            print(f"✅ Analysis completed successfully!", flush=True)
            print(f"{'='*60}\n", flush=True)
            
        except Exception as e:
            raise e
        
        # Look for the generated JSON file (main_pose.py creates it)
        # Check multiple possible locations
        possible_json_paths = [
            output_path.replace('.mp4', '_analysis.json'),
            output_path.replace('.avi', '_analysis.json'),
            os.path.join(OUTPUT_FOLDER, os.path.basename(output_path).replace('.mp4', '_analysis.json')),
            json_output_path
        ]
        
        analysis_data = None
        for json_path in possible_json_paths:
            if os.path.exists(json_path):
                print(f"✅ Found analysis JSON: {json_path}")
                with open(json_path, 'r') as f:
                    analysis_data = json.load(f)
                
                # Copy to session-specific location
                if json_path != json_output_path:
                    with open(json_output_path, 'w') as f:
                        json.dump(analysis_data, f, indent=2)
                    print(f"✅ Copied to: {json_output_path}")
                break
        
        # Extract stats from analysis data
        stats = {}
        if analysis_data:
            stats = {
                'player_stats': analysis_data.get('player_stats', {}),
                'rallies': len(analysis_data.get('rallies', [])),
                'bounces': len(analysis_data.get('bounces', [])),
                'score': analysis_data.get('score', {}),
                'match_summary': analysis_data.get('match_summary', {})
            }
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'complete',
            'message': 'Complete analysis finished! Video ready for download.',
            'progress': 100,
            'output_file': os.path.basename(output_path),
            'stats': stats,
            'terminal_output': terminal_output[:500]  # First 500 chars
        })
        
        processing_status[session_id] = 'complete'
        print(f"\n{'='*60}", flush=True)
        print(f"✅ ANALYSIS COMPLETE!", flush=True)
        print(f"📊 Session: {session_id}", flush=True)
        print(f"🎬 Video: {output_path}", flush=True)
        print(f"📄 JSON: {json_output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error processing video: {error_details}")
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'error',
            'message': f'Error: {str(e)}',
            'progress': 0,
            'error_details': error_details
        })
        processing_status[session_id] = 'error'


@app.route('/api/process-local', methods=['POST'])
def process_local_file():
    """Process a local file without uploading"""
    data = request.get_json()
    file_path = data.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path', 'success': False}), 400
    
    print(f"\n📁 Processing local file: {file_path}", flush=True)
    
    # Generate output path
    timestamp = int(time.time())
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"analyzed_{name}_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Create session ID
    session_id = f"session_{timestamp}"
    processing_status[session_id] = 'queued'
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_with_updates,
        args=(file_path, output_path, session_id)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'Processing started (local file).'
    })


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    print(f"📥 Upload request received")
    print(f"   Files in request: {list(request.files.keys())}")
    print(f"   Form data: {list(request.form.keys())}")
    
    if 'video' not in request.files:
        print("❌ Error: No 'video' field in request.files")
        return jsonify({'error': 'No video file provided', 'success': False}), 400
    
    file = request.files['video']
    print(f"   File object: {file}")
    print(f"   Filename: {file.filename}")
    
    if file.filename == '':
        print("❌ Error: Empty filename")
        return jsonify({'error': 'No file selected', 'success': False}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        timestamp = int(time.time())
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(video_path)
        
        # Generate output path (MP4 for browser compatibility)
        output_filename = f"analyzed_{name}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Create session ID
        session_id = f"session_{timestamp}"
        processing_status[session_id] = 'queued'
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_with_updates,
            args=(video_path, output_path, session_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Video uploaded successfully. Processing started.'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get processing status"""
    status = processing_status.get(session_id, 'unknown')
    return jsonify({'session_id': session_id, 'status': status})


@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download processed video"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/preview/<filename>', methods=['GET'])
def preview_frame(filename):
    """Get a preview frame from processed video"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})
    
    return jsonify({'error': 'Could not generate preview'}), 404


@app.route('/api/analysis/<session_id>', methods=['GET'])
def get_analysis(session_id):
    """Get analysis JSON for a session"""
    # Try to find analysis JSON file
    json_filename = f"analyzed_{session_id}_analysis.json"
    json_path = os.path.join(OUTPUT_FOLDER, json_filename)
    
    # Also try alternative naming patterns
    if not os.path.exists(json_path):
        # Look for any JSON file with session_id in name
        import glob
        pattern = os.path.join(OUTPUT_FOLDER, f"*{session_id}*analysis.json")
        matches = glob.glob(pattern)
        if matches:
            json_path = matches[0]
    
    if os.path.exists(json_path):
        import json
        with open(json_path, 'r') as f:
            analysis_data = json.load(f)
        return jsonify({
            'success': True,
            'analysis': analysis_data
        })
    
    return jsonify({'error': 'Analysis not found', 'success': False}), 404


@app.route('/api/coach/analyze-match', methods=['POST'])
def analyze_match_with_coach():
    """Get mental coaching advice based on video analysis"""
    try:
        # Try to import coach
        from RAG_MentalCoach.coach.coach import MentalCoachChatbot
        coach = MentalCoachChatbot()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Mental coach not available: {str(e)}. Check API keys and configuration.'
        }), 503
    
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('query', 'How can I improve my mental game based on this match?')
    
    if not session_id:
        return jsonify({'error': 'session_id is required', 'success': False}), 400
    
    # Get analysis JSON
    json_filename = f"analyzed_{session_id}_analysis.json"
    json_path = os.path.join(OUTPUT_FOLDER, json_filename)
    
    # Try alternative patterns
    if not os.path.exists(json_path):
        import glob
        pattern = os.path.join(OUTPUT_FOLDER, f"*{session_id}*analysis.json")
        matches = glob.glob(pattern)
        if matches:
            json_path = matches[0]
    
    video_analysis = None
    if os.path.exists(json_path):
        import json
        with open(json_path, 'r') as f:
            video_analysis = json.load(f)
    else:
        # If no analysis file, try to get from request
        video_analysis = data.get('video_analysis')
        if not video_analysis:
            return jsonify({
                'error': 'Analysis not found. Please provide session_id or video_analysis data.',
                'success': False
            }), 404
    
    try:
        # Search knowledge base
        context_items = coach.search_pinecone(query)
        
        # Generate response with video analysis context
        response = coach.generate_response(
            query=query,
            context_items=context_items,
            session_id=session_id,
            video_analysis=video_analysis
        )
        
        # Extract key stats for response
        player_1_stats = video_analysis.get('player_stats', {}).get('1', {})
        player_2_stats = video_analysis.get('player_stats', {}).get('2', {})
        
        return jsonify({
            'success': True,
            'response': response,
            'video_stats': {
                'player_1': {
                    'errors': player_1_stats.get('errors', 0),
                    'winners': player_1_stats.get('winners', 0),
                    'points_won': player_1_stats.get('points_won', 0),
                    'avg_speed_kmh': player_1_stats.get('avg_shot_speed_kmh', 0)
                },
                'player_2': {
                    'errors': player_2_stats.get('errors', 0),
                    'winners': player_2_stats.get('winners', 0),
                    'points_won': player_2_stats.get('points_won', 0),
                    'avg_speed_kmh': player_2_stats.get('avg_shot_speed_kmh', 0)
                },
                'total_rallies': len(video_analysis.get('rallies', [])),
                'max_ball_speed_kmh': video_analysis.get('statistics', {}).get('max_ball_speed_kmh', 0)
            },
            'sources_count': len(context_items) if context_items else 0
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in match analysis coaching: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/daytona/execute', methods=['POST'])
def execute_daytona_code():
    """Execute Python code securely in a Daytona sandbox"""
    try:
        from daytona_integration import get_daytona_executor
        
        executor = get_daytona_executor()
        if not executor:
            return jsonify({
                'success': False,
                'error': 'Daytona not configured. Set DAYTONA_API_KEY environment variable.'
            }), 503
        
        data = request.get_json()
        code = data.get('code')
        sandbox_id = data.get('sandbox_id')  # Optional: reuse existing sandbox
        
        if not code:
            return jsonify({'error': 'Code is required', 'success': False}), 400
        
        # Execute code in sandbox
        result = executor.execute_code(code, sandbox_id)
        
        return jsonify({
            'success': result['exit_code'] == 0,
            'exit_code': result['exit_code'],
            'result': result['result'],
            'sandbox_id': result['sandbox_id'],
            'error': result['error']
        })
        
    except ImportError:
        return jsonify({
            'success': False,
            'error': 'Daytona SDK not installed. Install with: pip install daytona'
        }), 503
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error executing Daytona code: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/daytona/sandbox/<sandbox_id>', methods=['DELETE'])
def delete_daytona_sandbox(sandbox_id):
    """Delete a Daytona sandbox"""
    try:
        from daytona_integration import get_daytona_executor
        
        executor = get_daytona_executor()
        if not executor:
            return jsonify({
                'success': False,
                'error': 'Daytona not configured. Set DAYTONA_API_KEY environment variable.'
            }), 503
        
        success = executor.delete_sandbox(sandbox_id)
        
        return jsonify({
            'success': success,
            'message': 'Sandbox deleted' if success else 'Sandbox not found'
        })
        
    except ImportError:
        return jsonify({
            'success': False,
            'error': 'Daytona SDK not installed. Install with: pip install daytona'
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    import sys
    
    # Get port from environment variable or use default (5001 - safe port)
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("🎾 Tennis Analysis Server Starting...")
    print(f"📡 Server running on http://localhost:{port}")
    print("🔌 WebSocket enabled for real-time updates")
    
    try:
        socketio.run(app, host=host, port=port, debug=True)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {port} is already in use.")
            print(f"💡 Try one of these solutions:")
            print(f"   1. Kill the process using port {port}: lsof -ti:{port} | xargs kill -9")
            print(f"   2. Use a different port: PORT=5002 python backend_server.py")
            sys.exit(1)
        else:
            raise
