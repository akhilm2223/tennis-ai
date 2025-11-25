"""
Mental Coaching API - Query Enhancement & Analysis
Transforms raw user input into contextual coaching queries using Claude Haiku 4.5
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Mental Coaching API",
    description="Tennis AI Mental Coaching - Query Enhancement Service",
    version="1.0.0"
)

# Initialize OpenRouter client with OpenAI SDK
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============================================================================
# Request/Response Models
# ============================================================================

class VideoAnalysis(BaseModel):
    """Video analysis data from tennis match"""
    video_info: Dict[str, Any]
    score: Dict[str, int]
    rallies: list
    bounces: list
    player_stats: Dict[str, Any]
    match_summary: Optional[Dict[str, Any]] = None
    shot_timeline: list
    tactical_analysis: Dict[str, Any]
    mental_indicators: Dict[str, Any]
    coaching_insights: Dict[str, Any]


class MentalCoachingRequest(BaseModel):
    """Main API request for mental coaching analysis"""
    video_analysis: VideoAnalysis
    query: str
    personal_info: str  # Markdown format
    past_conversation: Optional[Dict[str, Any]] = None


class MentalCoachingResponse(BaseModel):
    """API response with video analysis and enhanced query"""
    video_analysis: str
    enhanced_query: str


# ============================================================================
# Helper Functions
# ============================================================================

def generate_video_analysis_with_claude(video_analysis: VideoAnalysis) -> str:
    """Generate AI-powered video analysis using Claude Haiku 4.5 (150-200 words)"""
    
    # Convert video analysis to JSON string for Claude
    video_json = json.dumps(video_analysis.model_dump(), indent=2)
    
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a 10-year expert tennis coach. Analyze this match and give a plain English 
summary - 200-250 words maximum. Talk like a real coach to a player, not a scientist.

Focus on:
- What Player 1 did WELL and what they STRUGGLED with
- What Player 2 did WELL and what they STRUGGLED with  
- The KEY MOMENTS that changed the match
- AREAS TO WATCH for next time
- What each player needs to work on

Use simple language. Skip the technical jargon. Be honest and direct. Help them understand 
the match like you're talking over coffee, not reading a report.

Video Analysis Data:
{video_json}

Your Analysis:"""
                }
            ],
            max_tokens=600,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating video analysis: {str(e)}")


def extract_personal_context(personal_info: str) -> str:
    """Extract player profile context from personal info"""
    return f"""
PLAYER PROFILE:
{personal_info}
"""


def extract_historical_context(past_conversation: Optional[Dict[str, Any]]) -> str:
    """Extract patterns from past conversation history"""
    return f"""
PAST COACHING HISTORY:
{past_conversation if past_conversation else "No previous sessions recorded."}
"""


def build_enhanced_query(
    user_query: str,
    video_context: str,
    personal_context: str,
    historical_context: str
) -> str:
    """Synthesize all contexts into a coherent enhanced query"""
    
    enhanced_query = f"""
USER'S QUESTION/CONCERN:
{user_query}

{video_context}

{personal_context}

{historical_context}

ANALYSIS REQUEST:
Based on all the above context, provide:
1. Analysis of what technically happened in the match
2. Psychological/mental interpretation of the player's performance
3. Connection between personal factors (anxiety, style, history) and match outcomes
4. Specific mental coaching recommendations
5. Actionable drills or techniques to address identified issues
6. References to successful past techniques if applicable

Focus on MindServe's core mission: Understanding how pressure affects this player's game 
and providing mental resilience training, not just technical corrections.
"""
    
    return enhanced_query


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/mental-coaching/analyze", response_model=MentalCoachingResponse)
async def analyze_mental_coaching(request: MentalCoachingRequest) -> MentalCoachingResponse:
    """
    Analyze tennis match video and generate enhanced coaching query
    
    Process:
    1. Generate AI-powered video analysis using Claude (150-200 words)
    2. Extract personal player information
    3. Extract historical patterns from past conversations
    4. Use AI analysis as video context for enhanced query (200-250 words)
    5. Return both video analysis and enhanced query
    
    Args:
        request: MentalCoachingRequest with video_analysis, query, personal_info, past_conversation
    
    Returns:
        MentalCoachingResponse with video_analysis and enhanced_query fields
    """
    
    try:
        # Generate AI-powered video analysis using Claude
        video_analysis_output = generate_video_analysis_with_claude(request.video_analysis)
        
        # Extract personal and historical contexts
        personal_context = extract_personal_context(request.personal_info)
        historical_context = extract_historical_context(request.past_conversation)
        
        # Build enhanced query using Claude-generated video analysis
        enhanced_query = build_enhanced_query(
            user_query=request.query,
            video_context=video_analysis_output,
            personal_context=personal_context,
            historical_context=historical_context
        )
        
        # Return response with both video analysis and enhanced query
        return MentalCoachingResponse(
            video_analysis=video_analysis_output,
            enhanced_query=enhanced_query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing coaching analysis: {str(e)}")


@app.post("/api/v1/mental-coaching/generate-insights")
async def generate_coaching_insights(request: MentalCoachingRequest) -> Dict[str, str]:
    """
    Generate detailed coaching insights using Claude Haiku 4.5 via OpenRouter
    
    This endpoint processes the enhanced query through Claude for AI-powered insights
    
    Args:
        request: MentalCoachingRequest with all required fields
    
    Returns:
        Dictionary with coaching_insights and recommendations
    """
    
    try:
        # Generate AI-powered video analysis
        video_analysis_output = generate_video_analysis_with_claude(request.video_analysis)
        
        # Extract personal and historical contexts
        personal_context = extract_personal_context(request.personal_info)
        historical_context = extract_historical_context(request.past_conversation)
        
        # Build enhanced query using Claude-generated video analysis
        enhanced_query = build_enhanced_query(
            user_query=request.query,
            video_context=video_analysis_output,
            personal_context=personal_context,
            historical_context=historical_context
        )
        
        # Call Claude Haiku 4.5 via OpenRouter for coaching insights
        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a 10-year expert mental tennis coach. Review this player's data and provide 
a concise, plain English coaching summary - 100-150 words maximum. Be direct and practical.

{enhanced_query}

IMPORTANT: Keep response under 150 words. Use simple language. Focus on the ONE key issue 
and the ONE immediate action they should take. No jargon. Be like a coach talking to a player, 
not a therapist writing a report."""
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        
        # Extract insights from response
        coaching_response = response.choices[0].message.content
        
        return {
            "video_analysis": video_analysis_output,
            "enhanced_query": enhanced_query,
            "coaching_insights": coaching_response,
            "model_used": "anthropic/claude-haiku-4.5",
            "tokens_used": f"Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating coaching insights: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Mental Coaching API",
        "openrouter_configured": bool(OPENROUTER_API_KEY)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🎾 Mental Coaching API Starting...")
    print(f"OpenRouter API Key Loaded: {bool(OPENROUTER_API_KEY)}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
