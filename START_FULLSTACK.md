# 🎾 Tennis Analysis Full-Stack Setup

## Quick Start

### 1. Start Backend Server (Python)
```bash
# Install backend dependencies
pip install -r backend_requirements.txt

# Start Flask server
python backend_server.py
```
Server will run on: http://localhost:6000

### 2. Start Frontend (React)
```bash
# Navigate to frontend
cd tennis-3d-viewer

# Install dependencies (if not done)
npm install

# Start React dev server
npm run dev
```
Frontend will run on: http://localhost:5173

### 3. Use the App
1. Open browser to http://localhost:5173
2. Click "📹 Upload Video" button
3. Select a tennis video file
4. Click "🚀 Analyze Video"
5. Watch real-time progress
6. Download analyzed video when complete!

## Features
- ✅ Video upload through web interface
- ✅ Real-time processing updates via WebSocket
- ✅ Progress bar showing analysis status
- ✅ Download processed video
- ✅ 3D Coach Q character
- ✅ Voice interaction

## Architecture
- **Backend**: Flask + SocketIO (Python)
- **Frontend**: React + Three.js
- **Analysis**: MediaPipe + YOLO + OpenCV
- **Communication**: REST API + WebSocket

Enjoy! 🎾
