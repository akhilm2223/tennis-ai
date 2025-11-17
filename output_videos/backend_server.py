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
    """Process video and send real-time updates via WebSocket"""
    
    # Prepare JSON analysis file
    json_output_path = output_path.replace('.mp4', '_analysis.json')
    analysis_data = {
        'video_info': {},
        'players': [],
        'ball_tracking': [],
        'bounces': [],
        'court_detection': {},
        'statistics': {}
    }
    
    try:
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'starting',
            'message': 'Initializing detectors...',
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
            'message': f'Processing {total_frames} frames...',
            'progress': 5,
            'total_frames': total_frames,
            'fps': fps
        })
        
        # Progress callback for frame-by-frame updates
        def on_frame_processed(frame_num, total, frame_image):
            progress = int((frame_num / total) * 90) + 5  # 5-95%
            
            # Encode frame as JPEG for preview
            frame_preview = None
            if frame_image is not None:
                # Resize for faster transmission
                small_frame = cv2.resize(frame_image, (640, 360))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_preview = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('processing_update', {
                'session_id': session_id,
                'status': 'processing',
                'message': f'Processing frame {frame_num}/{total}...',
                'progress': progress,
                'frame_num': frame_num,
                'total_frames': total,
                'frame_preview': f'data:image/jpeg;base64,{frame_preview}' if frame_preview else None
            })
        
        # Process video with callback and get analysis data
        result = main_with_callback(
            video_path=video_path,
            output_path=output_path,
            calibrate=True,
            use_pose=True,
            show_bbox=False,
            progress_callback=on_frame_processed
        )
        
        # Save analysis JSON
        if result:
            analysis_data['video_info'] = {
                'path': video_path,
                'output': output_path,
                'fps': result.get('fps', fps),
                'total_frames': result.get('frames_processed', total_frames),
                'duration_seconds': result.get('frames_processed', total_frames) / fps
            }
            
            analysis_data['statistics'] = {
                'max_ball_speed_kmh': result.get('max_ball_speed', 0),
                'max_player_speeds_kmh': result.get('max_player_speeds', {}),
                'total_bounces': len(result.get('bounces', [])),
                'court_detected': result.get('court_detected', False)
            }
            
            analysis_data['bounces'] = result.get('bounces', [])
            analysis_data['ball_tracking'] = result.get('ball_trajectory', [])
            
            # Write JSON file
            import json
            with open(json_output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            print(f"✅ Analysis JSON saved: {json_output_path}")
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'complete',
            'message': 'Processing complete!',
            'progress': 100,
            'output_file': os.path.basename(output_path),
            'stats': result
        })
        
        processing_status[session_id] = 'complete'
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing video: {error_details}")
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'error',
            'message': f'Error: {str(e)}',
            'progress': 0
        })
        processing_status[session_id] = 'error'


@app.route('/api/process-local', methods=['POST'])
def process_local_file():
    """Process a local file without uploading"""
    data = request.get_json()
    file_path = data.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path', 'success': False}), 400
    
    print(f"📁 Processing local file: {file_path}")
    
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


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    import sys
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 6000))
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
            print(f"   2. Use a different port: PORT=5001 python backend_server.py")
            sys.exit(1)
        else:
            raise
