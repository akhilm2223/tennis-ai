# Mental Coaching API - Architecture & Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MindServe Mental Coaching System                     │
└─────────────────────────────────────────────────────────────────────────────┘

                              FRONTEND (React/Vue)
                                      │
                                      ↓
                    ┌──────────────────────────────────┐
                    │  Mental Coaching API (FastAPI)   │
                    │  🚀 mental_coaching_api.py       │
                    └──────────┬───────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
   Input Processing    Query Enhancement      AI Processing
   ─────────────────   ──────────────────      ──────────────
   
   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
   │ Video Data   │   │ Context         │   │ Claude Haiku 4.5 │
   │ (JSON)       │──→│ Extraction      │──→│ via OpenRouter   │
   └──────────────┘   └─────────────────┘   └──────────────────┘
   
   ┌──────────────┐   ┌─────────────────┐       (OPTIONAL)
   │ User Query   │   │ Enhanced Query  │
   │ (Text)       │──→│ Synthesis       │
   └──────────────┘   └────────┬────────┘
                               │
   ┌──────────────┐            │
   │ Personal     │────────────┤
   │ Info         │            │ (200-250 words)
   │ (Markdown)   │            │
   └──────────────┘            ↓
                      ┌─────────────────────┐
   ┌──────────────┐   │ Enhanced Query      │
   │ Past         │──→│ + Context Data      │
   │ Conversation │   │ (Ready for Claude)  │
   │ (JSON)       │   └────────┬────────────┘
   └──────────────┘            │
                               ↓
                      ┌──────────────────────┐
                      │ Coaching Insights    │
                      │ + Recommendations    │
                      └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
    Database            TTS Service            Dashboard
    (Pinecone)          (ElevenLabs)            (Web UI)
    Vector Store        Voice Output            Display Results
```

## Data Flow Diagram

```
START: User asks "Why did I lose that rally?"
│
├─→ VIDEO ANALYSIS EXTRACTION
│   ├─ Rally details (shots, winner, reason)
│   ├─ Speed data (224 kmh → 115 kmh = confidence drop)
│   ├─ Player stats (errors, winners, speed consistency)
│   └─ Mental indicators (pressure moments, hesitation)
│
├─→ PERSONAL INFO EXTRACTION
│   ├─ Height: 6'2"
│   ├─ Anxiety: Moderate
│   ├─ Playing style: Baseline aggressive
│   ├─ Health history: Previous shoulder strain
│   └─ Mental characteristics: Slow recovery after mistakes
│
├─→ HISTORICAL CONTEXT EXTRACTION
│   ├─ Recurring issues: Pressure on break points
│   ├─ Successful techniques: Breathing exercises
│   ├─ Session count: 3 previous sessions
│   └─ Progress: "Making steady progress, needs more practice"
│
├─→ CONTEXT SYNTHESIS
│   ├─ Combine all 4 inputs
│   ├─ Structure into clear sections
│   ├─ Link physical data to psychological patterns
│   ├─ Reference historical successes
│   └─ Target 200-250 words
│
├─→ ENHANCED QUERY GENERATION
│   │
│   └─ OUTPUT: Enhanced 200-250 word query
│       "USER'S QUESTION: Why did I lose that rally?
│        
│        TECHNICAL CONTEXT:
│        You hit 224 kmh opening shot but made unforced error.
│        Speed dropped to 115 kmh on follow-up (48% reduction).
│        Rally was only 2 shots before error.
│        
│        PLAYER PROFILE:
│        Baseline player, 6'2", anxiety under pressure.
│        Previous shoulder strain (recovered).
│        Slow recovery after mistakes.
│        
│        PAST COACHING HISTORY:
│        3 previous sessions. Recurring: pressure on key points.
│        Works: Breathing exercises, positive self-talk.
│        Progress: Steady but needs consistency practice.
│        
│        ANALYSIS REQUEST: Provide mental coaching insights..."
│
├─→ OPTIONAL: CLAUDE ANALYSIS
│   ├─ Send enhanced query to Claude Haiku 4.5
│   ├─ Claude understands:
│   │  ├─ Technical situation (224 → 115 kmh)
│   │  ├─ Psychological pattern (anxiety on follow-ups)
│   │  ├─ Player profile (baseline, anxious)
│   │  └─ Historical context (breathing helped)
│   │
│   └─ Claude generates personalized response
│       "Your issue is mental, not technical. You have power (224 kmh
│        proves it) but doubt yourself on follow-ups. This creates
│        hesitation and errors. Use 4-4-4 breathing between points.
│        Practice 3-shot rally builds. This works for you (we've seen
│        it in past sessions)."
│
└─→ RETURN RESPONSE
    {
      "enhanced_query": "...",
      "coaching_insights": "Your issue is mental...",
      "model_used": "anthropic/claude-haiku-4.5",
      "tokens_used": "Input: 1250, Output: 750"
    }
```

## Component Interaction

```
┌──────────────────────────────────────────────────────────────────┐
│                    mental_coaching_api.py                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ FastAPI Application                                        │ │
│  │                                                            │ │
│  │  Endpoint 1: /api/v1/mental-coaching/analyze              │ │
│  │  → Returns enhanced query only                            │ │
│  │                                                            │ │
│  │  Endpoint 2: /api/v1/mental-coaching/generate-insights   │ │
│  │  → Returns enhanced query + Claude analysis               │ │
│  │                                                            │ │
│  │  Endpoint 3: /health                                      │ │
│  │  → Returns health status                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│  ┌──────────────────────┴──────────────────────────────────────┐ │
│  │ Context Extraction Functions                               │ │
│  │                                                            │ │
│  │  • extract_video_context()      → Rally, speed, stats     │ │
│  │  • extract_personal_context()   → Player profile          │ │
│  │  • extract_historical_context() → Past patterns           │ │
│  │  • build_enhanced_query()       → Synthesis               │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                         │
│  ┌────────────────────┴──────────────────────────────────────┐ │
│  │ OpenRouter Integration                                    │ │
│  │                                                            │ │
│  │  client = OpenAI(                                         │ │
│  │    base_url="https://openrouter.ai/api/v1",            │ │
│  │    api_key=OPENROUTER_API_KEY                           │ │
│  │  )                                                        │ │
│  │                                                            │ │
│  │  response = client.chat.completions.create(             │ │
│  │    model="anthropic/claude-haiku-4.5",                 │ │
│  │    messages=[...]                                        │ │
│  │  )                                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    .env File    Pydantic Models   System Prompts
    (config)     (Validation)      (Instructions)
```

## Integration Points with Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                    MENTAL COACHING API                          │
│              (Your new FastAPI application)                     │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ↓                    ↓                    ↓
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │ VIDEO ANALYSIS │  │ VECTOR DATABASE│  │  VOICE COACH   │
   │ JSON Parser    │  │  (Pinecone)    │  │  (ElevenLabs)  │
   │                │  │                │  │                │
   │ Input:         │  │ Fetch:         │  │ Input:         │
   │ - Rally data   │  │ Similar past   │  │ - Insights     │
   │ - Speeds       │  │   sessions     │  │                │
   │ - Errors       │  │ - Advice       │  │ Output:        │
   │                │  │ - Drills       │  │ - Audio stream │
   │ Output:        │  │                │  │ - MP3 file     │
   │ - Context      │  │ Search using:  │  │                │
   │ - Metrics      │  │ - Enhanced     │  │ Reads:         │
   │                │  │   query        │  │ - Insights     │
   │                │  │ - Keywords     │  │ - Coach voice  │
   │                │  │                │  │ - Tone/speed   │
   └────────────────┘  └────────────────┘  └────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ↓                     ↓
            ┌─────────────────┐  ┌─────────────────┐
            │ FRONTEND DISPLAY│  │ PLAYER TRAINING │
            │ (Dashboard)     │  │ (Mobile App)    │
            │                 │  │                 │
            │ Display:        │  │ Presents:       │
            │ - Insights      │  │ - Drills        │
            │ - Stats         │  │ - Advice        │
            │ - Video frames  │  │ - Motivation    │
            │ - Timeline      │  │ - Progress      │
            └─────────────────┘  └─────────────────┘
```

## Environment Variable Flow

```
.env File
  │
  └─→ OPENROUTER_API_KEY = sk-or-v1-...
      ANTHROPIC_API_KEY = sk-ant-...
      PINECONE_API_KEY = pcsk_...
      ELEVENLABS_API_KEY = sk_...
      GALILEO_API_KEY = ...
      │
      ├─→ load_dotenv() in mental_coaching_api.py
      │   │
      │   ├─→ os.getenv("OPENROUTER_API_KEY")
      │   │   │
      │   │   └─→ OpenAI(api_key=OPENROUTER_API_KEY)
      │   │       │
      │   │       └─→ client.chat.completions.create(
      │   │           model="anthropic/claude-haiku-4.5"
      │   │       )
      │   │
      │   ├─→ Settings.validate()
      │   │   │
      │   │   └─→ Raises error if key missing
      │   │
      │   └─→ config.py uses all keys
      │       (Pinecone, ElevenLabs, Galileo, etc.)
      │
      └─→ Application starts with full configuration
```

## Request Processing Pipeline

```
HTTP Request
     │
     ↓
  ┌─────────────────────────────────────┐
  │ Pydantic Validation                 │
  │ - Check all required fields         │
  │ - Validate data types               │
  │ - Convert JSON to Python objects    │
  └──────────────┬──────────────────────┘
                 │ (Valid)
                 ↓
  ┌─────────────────────────────────────┐
  │ Context Extraction                  │
  │ - Extract video metrics             │
  │ - Parse personal info               │
  │ - Identify historical patterns      │
  │ - Preserve user query emotion       │
  └──────────────┬──────────────────────┘
                 │
                 ↓
  ┌─────────────────────────────────────┐
  │ Enhanced Query Building             │
  │ - Combine all contexts              │
  │ - Structure with headers            │
  │ - Target 200-250 words              │
  │ - Focus on mental coaching angle    │
  └──────────────┬──────────────────────┘
                 │
         ┌───────┴───────┐
         │ (Conditional)  │
         ↓                ↓
    Return Query    Call Claude
     Response       via OpenRouter
         │                │
         └────────┬───────┘
                  │
                  ↓
         ┌──────────────────┐
         │ JSON Response    │
         │ (Structured)     │
         └────────┬─────────┘
                  │
                  ↓
          HTTP Response 200 OK
```

## Error Handling Flow

```
Request Received
      │
      ├─→ Validation Error
      │   └─→ HTTPException 422
      │       {"detail": "Invalid field type"}
      │
      ├─→ Missing API Key
      │   └─→ ValueError
      │       "OPENROUTER_API_KEY not found"
      │
      ├─→ OpenRouter Connection Error
      │   └─→ HTTPException 503
      │       "External service unavailable"
      │
      ├─→ Claude Response Error
      │   └─→ HTTPException 500
      │       "Error generating coaching insights"
      │
      └─→ Success
          └─→ 200 OK
              {"enhanced_query": "...", "coaching_insights": "..."}
```

---

This architecture ensures:
✅ Clean separation of concerns
✅ Secure API key handling
✅ Flexible input processing
✅ Optional AI integration
✅ Clear error handling
✅ Ready for scaling
