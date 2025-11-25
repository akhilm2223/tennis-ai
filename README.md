# 🎾 MindServe - AI Tennis Analysis & Mental Coaching

**MindServe** is an intelligent tennis analysis system that combines computer vision with AI-powered mental coaching to help players improve both their tactical game and mental performance.

## ✨ Key Features

### 📹 Advanced Video Analysis
- **EXACT Court Line Detection** - Manually calibrated court lines for pixel-perfect accuracy
- **Ball Tracking** - Ultra-persistent tracking with Kalman filter prediction
- **Player Pose Tracking** - MediaPipe-based skeleton tracking for both players
- **Mini-Court Visualization** - Real-time tactical overview showing ball position

### 📊 Comprehensive Coaching Data
MindServe generates a detailed JSON output with everything a tennis coach needs:

- **Shot Timeline** - Every shot with timestamp, player, position, speed, and pressure level
- **Tactical Analysis** - Court usage patterns, shot distribution, pressure situations
- **Mental Indicators** - Confidence drops, pressure moments, recovery patterns
- **Coaching Insights** - AI-ready structure for strengths, weaknesses, and recommendations

### 🧠 AI Mental Coaching
- RAG-based mental coaching system with professional tennis psychology content
- Books, articles, podcasts, and scientific papers on tennis mental performance
- Real-time chat interface for personalized mental coaching advice

### 🎯 3D Visualization
- Interactive 3D court viewer with ball trajectory
- Upload and analyze your own match videos
- Visual tactical insights

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd tennis-3d-viewer && npm install
```

### 2. Run Video Analysis
```bash
python main_pose.py --video "your_match.mov"
```

This will generate:
- **Video output** with court lines, ball tracking, and player skeletons
- **Coaching JSON** with comprehensive match analysis data

### 3. Start Full Application
```bash
# Terminal 1: Backend
python backend_server.py

# Terminal 2: Frontend
cd tennis-3d-viewer
npm run dev
```

Then open: `http://localhost:5173`

## 📁 Project Structure

```
tennis-ai-main/
├── main_pose.py                 # Main analysis script
├── court_lines_manual.json      # Court calibration
├── backend_server.py            # Flask API server
├── trackers/                    # Ball & court tracking
├── modules/                     # Pose detection
├── RAG_MentalCoach/            # AI coaching system
├── tennis-3d-viewer/           # React frontend
└── output_videos/              # Analysis results
```

## 🎾 Core Output Files

1. **Video**: `output_videos/tennis_analysis_trail.avi` - Annotated match video
2. **JSON**: `output_videos/tennis_analysis_trail.json` - Coaching data

### JSON Structure
```json
{
  "shot_timeline": [/* Shot-by-shot breakdown */],
  "tactical_analysis": {/* Court usage, shot distribution */},
  "mental_indicators": {/* Confidence, pressure, recovery */},
  "coaching_insights": {/* Strengths, weaknesses, recommendations */}
}
```

## 🛠️ Technical Stack

- **Computer Vision**: OpenCV, YOLO (ball detection), MediaPipe (pose)
- **Tracking**: Kalman Filter, Physics-based prediction
- **Backend**: Flask, SocketIO
- **Frontend**: React, Three.js, Vite
- **AI**: RAG system with vector embeddings

## 📝 Key Commands

**Analyze a match:**
```bash
python main_pose.py --video "match.mov"
```

**Calibrate court lines:**
```bash
python manual_court_lines_full.py
```

**Verify court lines:**
```bash
python visualize_manual_lines.py --video "match.mov" --lines "court_lines_manual.json"
```

## 🎯 Use Cases

- **Players**: Analyze your matches, understand tactical patterns, get mental coaching
- **Coaches**: Generate detailed match reports, identify player strengths/weaknesses
- **Analysts**: Extract comprehensive match data for tactical insights

## 🏆 What Makes MindServe Unique

Unlike simple stat trackers, MindServe provides:
- ✅ Shot-by-shot tactical analysis with context
- ✅ Mental performance indicators (confidence, pressure handling)
- ✅ AI-ready coaching insights for personalized advice
- ✅ Pixel-perfect court line accuracy
- ✅ Professional-grade tracking algorithms

Perfect for creating an AI tennis coach that goes beyond statistics to provide real tactical and mental coaching.

---

**MindServe** - Where mental coaching meets tennis intelligence. 🎾🧠

