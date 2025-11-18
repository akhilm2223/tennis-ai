"""
Galileo MCP Service
Handles integration with Galileo AI observability and evaluation platform
"""

import os
import json
from typing import Optional, Dict, Any, List

# Global state
_galileo_client = None
_galileo_available = False
_galileo_api_key = None


def initialize_galileo():
    """Initialize Galileo MCP client"""
    global _galileo_client, _galileo_available, _galileo_api_key
    
    try:
        # Try to import Galileo SDK if available
        try:
            from galileo_sdk import Galileo
            _galileo_client = Galileo
        except ImportError:
            # If SDK not available, we'll use direct API calls
            _galileo_client = None
        
        _galileo_api_key = os.getenv("GALILEO_API_KEY")
        if _galileo_api_key:
            _galileo_available = True
            print("✅ Galileo MCP initialized")
            return True
        else:
            _galileo_available = False
            _galileo_client = None
            print("⚠️  GALILEO_API_KEY not found. Galileo observability disabled.")
            return False
    except Exception as e:
        _galileo_available = False
        _galileo_client = None
        print(f"⚠️  Warning: Could not initialize Galileo: {e}")
        return False


def is_available():
    """Check if Galileo MCP is available"""
    return _galileo_available


def log_event(
    event_type: str,
    data: Dict[str, Any],
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log an event to Galileo
    
    Args:
        event_type: Type of event (e.g., 'coach_query', 'video_analysis', 'tts_generation')
        data: Event data dictionary
        session_id: Optional session identifier
        metadata: Optional metadata dictionary
    
    Returns:
        bool: True if logged successfully, False otherwise
    """
    if not _galileo_available or not _galileo_api_key:
        return False
    
    try:
        import requests
        
        # Galileo API endpoint (adjust based on actual API)
        api_url = os.getenv("GALILEO_API_URL", "https://api.galileo.ai/v1/events")
        
        payload = {
            "event_type": event_type,
            "data": data,
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
        headers = {
            "Authorization": f"Bearer {_galileo_api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Galileo log event error: {e}")
        return False


def log_coach_interaction(
    query: str,
    response: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    response_time_ms: Optional[float] = None
):
    """
    Log a mental coach interaction to Galileo
    
    Args:
        query: User query/question
        response: Coach response
        sources: List of sources used in response
        session_id: Session identifier
        response_time_ms: Response time in milliseconds
    
    Returns:
        bool: True if logged successfully
    """
    data = {
        "query": query,
        "response": response,
        "response_length": len(response),
        "query_length": len(query),
        "sources_count": len(sources) if sources else 0
    }
    
    if response_time_ms is not None:
        data["response_time_ms"] = response_time_ms
    
    if sources:
        data["sources"] = sources[:5]  # Limit to first 5 sources
    
    metadata = {
        "service": "mental_coach",
        "has_sources": sources is not None and len(sources) > 0
    }
    
    return log_event("coach_interaction", data, session_id, metadata)


def log_video_analysis(
    video_path: str,
    analysis_data: Dict[str, Any],
    session_id: Optional[str] = None,
    processing_time_seconds: Optional[float] = None
):
    """
    Log video analysis results to Galileo
    
    Args:
        video_path: Path to analyzed video
        analysis_data: Analysis results dictionary
        session_id: Session identifier
        processing_time_seconds: Processing time in seconds
    
    Returns:
        bool: True if logged successfully
    """
    data = {
        "video_path": video_path,
        "analysis": analysis_data,
        "has_ball_tracking": "ball_tracking" in analysis_data,
        "has_player_tracking": "players" in analysis_data,
        "has_court_detection": "court_detection" in analysis_data
    }
    
    if processing_time_seconds is not None:
        data["processing_time_seconds"] = processing_time_seconds
    
    metadata = {
        "service": "video_analysis",
        "video_name": os.path.basename(video_path) if video_path else None
    }
    
    return log_event("video_analysis", data, session_id, metadata)


def log_tts_generation(
    text: str,
    voice_id: str,
    audio_length_bytes: Optional[int] = None,
    generation_time_ms: Optional[float] = None
):
    """
    Log TTS generation to Galileo
    
    Args:
        text: Text that was converted to speech
        voice_id: Voice ID used
        audio_length_bytes: Size of generated audio in bytes
        generation_time_ms: Generation time in milliseconds
    
    Returns:
        bool: True if logged successfully
    """
    data = {
        "text": text[:500],  # Truncate for logging
        "text_length": len(text),
        "voice_id": voice_id,
        "truncated": len(text) > 500
    }
    
    if audio_length_bytes is not None:
        data["audio_length_bytes"] = audio_length_bytes
    
    if generation_time_ms is not None:
        data["generation_time_ms"] = generation_time_ms
    
    metadata = {
        "service": "tts",
        "provider": "elevenlabs"
    }
    
    return log_event("tts_generation", data, None, metadata)


def evaluate_response(
    query: str,
    response: str,
    context: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a response using Galileo's evaluation tools
    
    Args:
        query: Original query
        response: Generated response
        context: Optional context items
        metrics: Optional list of metrics to evaluate (e.g., ['relevance', 'coherence'])
    
    Returns:
        Dictionary with evaluation results, or None if unavailable
    """
    if not _galileo_available or not _galileo_api_key:
        return None
    
    try:
        import requests
        
        api_url = os.getenv("GALILEO_EVAL_API_URL", "https://api.galileo.ai/v1/evaluate")
        
        payload = {
            "query": query,
            "response": response,
            "context": context or [],
            "metrics": metrics or ["relevance", "coherence", "helpfulness"]
        }
        
        headers = {
            "Authorization": f"Bearer {_galileo_api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        print(f"⚠️  Galileo evaluation error: {e}")
        return None


def get_client():
    """Get the Galileo client instance (for advanced usage)"""
    return _galileo_client


def get_api_key():
    """Get the Galileo API key (for debugging)"""
    return _galileo_api_key if _galileo_available else None
