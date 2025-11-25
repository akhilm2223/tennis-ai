# Postman Testing Guide - Mental Coaching API

## 📋 Overview

This guide shows you how to test the Mental Coaching API using Postman.

**Key Change**: The API now calls Claude Haiku 4.5 to analyze the video data (150-200 words) before generating the enhanced query. Both outputs are returned.

---

## 🚀 Setup Steps

### Step 1: Import Postman Collection

1. **Open Postman** (download from https://www.postman.com/downloads/ if needed)
2. **Click "Import"** button (top left)
3. **Select file**: `MentalCoachingAPI.postman_collection.json` (in your backend folder)
4. **Click "Import"** button
5. ✅ You should see 3 requests loaded:
   - Health Check
   - Analyze Mental Coaching (Enhanced Query Only)
   - Generate Coaching Insights (Full Analysis)

### Step 2: Start the API

```bash
# In your terminal, navigate to backend folder
cd d:\Projects\tennis-ai\backend

# Start the API
python mental_coaching_api.py

# You should see:
# 🎾 Mental Coaching API Starting...
# OpenRouter API Key Loaded: True
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Configure Postman Environment (Optional)

1. Click **"Environments"** in top left
2. Click **"+"** to create new environment
3. Name it: `Mental Coaching Dev`
4. Add variable:
   ```
   base_url = http://localhost:8000
   ```
5. Select this environment from dropdown

Then replace hardcoded URLs with `{{base_url}}/api/v1/mental-coaching/analyze`

---

## 🧪 Testing the 3 Endpoints

### Test 1: Health Check

**Purpose**: Verify API is running

**Steps**:
1. Click the **"Health Check"** request
2. Click **"Send"** button
3. You should see response:
   ```json
   {
     "status": "healthy",
     "service": "Mental Coaching API",
     "openrouter_configured": true
   }
   ```

**Expected**: Status code `200 OK` with `openrouter_configured: true`

---

### Test 2: Analyze Mental Coaching (Enhanced Query Only)

**Purpose**: Get both AI-generated video analysis AND enhanced query

**What it does**:
1. Takes video analysis JSON
2. Sends to Claude Haiku 4.5 (via OpenRouter)
3. Claude generates 150-200 word analysis (what players did right/wrong, crucial moments)
4. Uses that analysis to build enhanced query (200-250 words)
5. Returns BOTH outputs

**Steps**:
1. Click the **"Analyze Mental Coaching (Enhanced Query Only)"** request
2. Review the request body (JSON with video data, player query, personal info, past conversation)
3. Click **"Send"** button
4. Wait 10-15 seconds (Claude is analyzing)

**Expected Response** (Status: `200 OK`):
```json
{
  "video_analysis": "Player 1 started aggressively with a 224 kmh opening shot, demonstrating technical prowess and initial confidence. However, a significant 49% speed reduction to 115 kmh on the follow-up indicates a mental collapse post-miss. This pattern suggests hesitation replacing aggression. Player 2 responded methodically with 207.5 kmh, capitalizing on Player 1's unforced error. The crucial moment was Player 1's inability to finish despite establishing dominance. Mental indicators show confidence drops correlating with the speed degradation. Player 1's strength (power) became a weakness when doubt creept in after the initial miss. Recovery patterns suggest slow psychological rebound needed addressing...",
  
  "enhanced_query": "USER'S QUESTION/CONCERN:\nWhy did I lose that opening rally despite hitting a powerful shot?\n\nAI-GENERATED VIDEO ANALYSIS:\nPlayer 1 started aggressively... [full analysis]\n\nPLAYER PROFILE:\nHeight: 6'2\", Anxiety: Moderate...\n\nPAST COACHING HISTORY:\nSession count: 3, Recurring: Pressure on break points...\n\nANALYSIS REQUEST:\nBased on all the above context, provide mental coaching insights..."
}
```

**What to look for**:
- ✅ `video_analysis` field contains Claude's 150-200 word analysis
- ✅ `enhanced_query` field contains 200-250 word synthesized prompt
- ✅ Analysis mentions "what player did right/wrong"
- ✅ Analysis mentions speed drop (224→115 kmh)
- ✅ Analysis identifies pressure/mental factors

---

### Test 3: Generate Coaching Insights (Full Analysis)

**Purpose**: Get video analysis + enhanced query + Claude's coaching recommendations

**What it does**:
1. Same as Test 2, PLUS
2. Sends enhanced query to Claude again
3. Claude generates personalized mental coaching insights (1500 tokens max)
4. Returns ALL outputs

**Steps**:
1. Click the **"Generate Coaching Insights (Full Analysis)"** request
2. Click **"Send"** button
3. Wait 20-30 seconds (Claude analyzes twice)

**Expected Response** (Status: `200 OK`):
```json
{
  "video_analysis": "Player 1 started aggressively... [150-200 word analysis]",
  
  "enhanced_query": "USER'S QUESTION/CONCERN:... [200-250 word query]",
  
  "coaching_insights": "Your issue is psychological, not technical. You demonstrated significant power (224 kmh) but allowed doubt to creep in after one miss. The 49% speed reduction shows your mental state shifted from aggressive to defensive. This is a classic pressure response: initial confidence followed by hesitation.\n\nKey observation: You lost not because you're weak, but because pressure made you doubt your shot. This is addressable.\n\nImmediate action items:\n1. Mental Reset: Use your proven 4-4-4 breathing between points (you've practiced this)\n2. Confidence Building: After every strong shot, visualize it landing successfully\n3. Point Strategy: Build points with 2-3 hits before attempting winners\n\nDrill for this week: Play 20 baseline rallies focusing on consistency first, winners second.",
  
  "model_used": "anthropic/claude-haiku-4.5",
  
  "tokens_used": "Input: 1250, Output: 850"
}
```

**What to look for**:
- ✅ All 4 fields present: `video_analysis`, `enhanced_query`, `coaching_insights`, `tokens_used`
- ✅ `coaching_insights` contains actionable mental coaching (not technical tips)
- ✅ References player's history ("4-4-4 breathing you've practiced")
- ✅ Identifies mental issue (pressure, not technique)
- ✅ Provides specific drills

---

## 🔍 Understanding the Response Flow

```
Test 2 Response (Analyze):
┌─────────────────────────────────────────┐
│ video_analysis (150-200 words)          │ ← Claude's analysis of video data
│ AI-generated match analysis             │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ enhanced_query (200-250 words)          │ ← Enhanced query built from analysis
│ User question + video analysis +        │   + personal info + history
│ personal info + history                 │
└─────────────────────────────────────────┘


Test 3 Response (Generate Insights):
┌─────────────────────────────────────────┐
│ video_analysis (150-200 words)          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ enhanced_query (200-250 words)          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ coaching_insights (1500 tokens)         │ ← Claude's mental coaching
│ Personalized recommendations            │   based on enhanced query
│ Specific drills & techniques            │
└─────────────────────────────────────────┘
```

---

## 📝 How to Modify Test Data

### Change the Player Query
In the request body, find:
```json
"query": "Why did I lose that opening rally despite..."
```
Replace with your own question.

### Add More Rally Data
In `rallies` array, add more entries:
```json
"rallies": [
  {...existing},
  {
    "rally_number": 2,
    "start_frame": 100,
    "end_frame": 150,
    "shots": 5,
    "point_winner": 1,
    "reason": "winner",
    "max_ball_speed_kmh": 198.5
  }
]
```

### Update Personal Info
Modify the `personal_info` markdown:
```
# Player Profile

## Anxiety Level
- Severe (instead of Moderate)

## Playing Style
- Serve & volley (instead of Baseline aggressive)
```

### Add New Past Techniques
In `past_conversation`:
```json
"successful_techniques": [
  "Existing technique",
  "New technique - Visualization of successful serves"
]
```

---

## 🐛 Troubleshooting

### Error: "Connection refused"
**Problem**: API not running
**Solution**: 
```bash
python mental_coaching_api.py
```

### Error: "OPENROUTER_API_KEY not found"
**Problem**: .env file missing or key not set
**Solution**: Ensure `.env` file in backend folder has:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### Error: "timeout" (request takes >60 seconds)
**Problem**: Claude is slow or taking multiple calls
**Solution**: This is normal for Test 3. Increase Postman timeout:
1. Click request → **"Tests"** tab
2. Or set global timeout: **Settings** → **Timeout** → Set to `120000` (ms)

### Error: "422 Unprocessable Entity"
**Problem**: JSON format error
**Solution**: 
1. Right-click response body
2. Select **"Pretty Print"** to see formatting
3. Check all fields match the schema

### Response is empty or null
**Problem**: Claude API error
**Solution**: 
1. Check `.env` file has correct OPENROUTER_API_KEY
2. Verify API key is active (visit https://openrouter.ai/)
3. Check rate limiting (you might be hitting API limits)

---

## ✨ What Changed from Previous Version

| Feature | Before | After |
|---------|--------|-------|
| **Video Analysis** | Parsed JSON fields manually | Claude generates 150-200 word AI analysis |
| **Response** | Only `enhanced_query` | Both `video_analysis` + `enhanced_query` |
| **Insights Endpoint** | 2 fields returned | 5 fields returned (+ video_analysis, + tokens_used) |
| **Analysis Quality** | Generic extraction | AI-powered contextual analysis |

---

## 🎯 Example: Full Testing Workflow

```
1. Start API
   $ python mental_coaching_api.py

2. Health Check (Postman)
   GET http://localhost:8000/health
   Expected: {"status": "healthy", ...}

3. Analyze Request (Postman)
   POST http://localhost:8000/api/v1/mental-coaching/analyze
   Body: {...video data, query, personal info...}
   Response: {"video_analysis": "...", "enhanced_query": "..."}

4. Review Video Analysis
   Check Claude's 150-200 word analysis of match
   Verify it identifies player strengths/weaknesses

5. Review Enhanced Query
   Verify it incorporates video analysis + personal + history
   Check length is ~200-250 words

6. Generate Insights (Postman)
   POST http://localhost:8000/api/v1/mental-coaching/generate-insights
   Same body as step 3
   Response: {...video_analysis, enhanced_query, coaching_insights, tokens_used}

7. Review Coaching Insights
   Check recommendations are mental-focused (not technique)
   Verify they reference player's history
   Ensure specific drills are provided
```

---

## 🎾 Success Criteria

✅ Health check returns healthy status
✅ Analyze endpoint returns video_analysis (Claude-generated)
✅ Analyze endpoint returns enhanced_query (200-250 words)
✅ Insights endpoint returns coaching_insights (personalized mental coaching)
✅ All responses take 10-30 seconds (API calls to Claude)
✅ Video analysis identifies what players did right and wrong
✅ Coaching insights reference player's history
✅ Drills are specific and actionable

---

**Your API is now ready to deliver personalized mental coaching powered by Claude Haiku 4.5!** 🧠🎾
