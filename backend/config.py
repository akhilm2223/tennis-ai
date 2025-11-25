"""
Configuration module for Mental Coaching API
Centralized environment and settings management
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "anthropic/claude-haiku-4.5"
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    
    # API Configuration
    API_TITLE: str = "Mental Coaching API"
    API_DESCRIPTION: str = "Tennis AI Mental Coaching - Query Enhancement Service"
    API_VERSION: str = "1.0.0"
    
    # Server Configuration
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    SERVER_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Claude Configuration
    CLAUDE_MAX_TOKENS: int = 1500
    CLAUDE_TEMPERATURE: float = 0.7
    
    # Enhanced Query Configuration
    ENHANCED_QUERY_MIN_LENGTH: int = 200
    ENHANCED_QUERY_MAX_LENGTH: int = 250
    
    # Request Timeouts (in seconds)
    REQUEST_TIMEOUT: int = 60
    HEALTH_CHECK_TIMEOUT: int = 10
    
    # Galileo Configuration (for observability)
    GALILEO_API_KEY: Optional[str] = os.getenv("GALILEO_API_KEY", None)
    GALILEO_PROJECT_NAME: Optional[str] = os.getenv("GALILEO_PROJECT_NAME", None)
    GALILEO_LOG_STREAM: Optional[str] = os.getenv("GALILEO_LOG_STREAM", None)
    
    # Pinecone Configuration (for vector database)
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY", None)
    PINECONE_INDEX: Optional[str] = os.getenv("PINECONE_INDEX", "mental-coaching")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please check your .env file."
            )
        
        return True
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get a summary of current configuration (without sensitive data)"""
        
        return {
            "api_title": cls.API_TITLE,
            "api_version": cls.API_VERSION,
            "server_host": cls.SERVER_HOST,
            "server_port": cls.SERVER_PORT,
            "openrouter_configured": bool(cls.OPENROUTER_API_KEY),
            "anthropic_configured": bool(cls.ANTHROPIC_API_KEY),
            "galileo_configured": bool(cls.GALILEO_API_KEY),
            "pinecone_configured": bool(cls.PINECONE_API_KEY),
            "claude_model": cls.OPENROUTER_MODEL,
            "max_tokens": cls.CLAUDE_MAX_TOKENS,
            "temperature": cls.CLAUDE_TEMPERATURE,
        }


# System Prompts

MENTAL_COACH_SYSTEM_PROMPT = """You are an expert mental tennis coach specializing in MindServe - 
building mental resilience, confidence, and psychological performance in tennis players under pressure.

Your approach focuses on:
1. Identifying psychological patterns in performance
2. Understanding how pressure affects decision-making
3. Providing evidence-based mental training techniques
4. Building personalized coaching recommendations
5. Connecting physical performance to mental state

Key Principles:
- Mental coaching is about building resilience, not just fixing problems
- Pressure is inevitable; the key is managing your response to it
- Confidence comes from preparation and controlled practice
- Recovery after mistakes is a learnable skill
- Small mental shifts create big performance changes

Remember: Tennis is 90% mental. Your job is to help players train that 90%."""

ENHANCED_QUERY_TEMPLATE = """
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

# Response Templates

SUCCESS_RESPONSE_TEMPLATE = {
    "status": "success",
    "data": {
        "enhanced_query": None,
        "coaching_insights": None,
        "model_used": None,
        "tokens_used": None,
        "recommendations": None
    }
}

ERROR_RESPONSE_TEMPLATE = {
    "status": "error",
    "error": {
        "code": None,
        "message": None,
        "details": None
    }
}

# Logging Configuration

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
        },
        "file": {
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "mental_coaching_api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
    },
    "loggers": {
        "mental_coaching_api": {
            "handlers": ["default", "file"],
            "level": "INFO",
        }
    }
}


# Initialize and validate settings
def initialize_settings():
    """Initialize settings and validate configuration"""
    try:
        Settings.validate()
        print("✅ Settings validated successfully")
        print(f"📊 Configuration Summary:")
        summary = Settings.get_summary()
        for key, value in summary.items():
            print(f"   - {key}: {value}")
        return True
    except ValueError as e:
        print(f"❌ Configuration Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Test settings initialization
    initialize_settings()
