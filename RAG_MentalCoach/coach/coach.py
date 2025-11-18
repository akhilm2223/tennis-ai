import os
import dotenv

from pinecone import Pinecone
import google.generativeai as genai  # Still used for embeddings
from anthropic import Anthropic  # For Claude 4 Sonnet

dotenv.load_dotenv()

# -------------------------
# Constants
# -------------------------
INDEX_NAME = "mental-coaching"
TOP_K = 10
RELEVANCE_THRESHOLD = 0.6
MAX_CONTEXT_SOURCES = 10


# -------------------------
# API Keys
# -------------------------
# Gemini API key (still used for embeddings)
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env (needed for embeddings)")

genai.configure(api_key=google_api_key)

# Anthropic API key (for Claude Haiku 4.5)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")

anthropic_client = Anthropic(api_key=anthropic_api_key)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_api_key)
knowledge_index = pinecone_client.Index(INDEX_NAME)


# ============================================================
#   MENTAL COACH RAG CHATBOT
# ============================================================

class MentalCoachChatbot:
    """RAG chatbot for mental coaching using Pinecone + Claude Haiku 4.5."""

    def __init__(self):
        # Use Claude 3.5 Haiku for text generation (faster and more cost-effective)
        self.model_name = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku
        self.client = anthropic_client

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------
    def generate_query_embedding(self, query: str):
        """Generate embedding using text-embedding-004."""
        try:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return res.get("embedding")
        except Exception as e:
            print("Embedding error:", e)
            return None

    # --------------------------------------------------------
    # Pinecone Search
    # --------------------------------------------------------
    def search_pinecone(self, query: str, top_k=TOP_K):
        """
        Search Pinecone for relevant context.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
        """
        vec = self.generate_query_embedding(query)
        if vec is None:
            return []

        try:
            res = knowledge_index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True
            )
            matches = res.matches if hasattr(res, "matches") else res.get("matches", [])
            filtered = self._filter_and_sort_matches(matches)
            
            return filtered
        except Exception as e:
            print("Pinecone error:", e)
            return []

    def _filter_and_sort_matches(self, matches):
        """Filter by relevance and return the top context items."""
        items = []
        for m in matches:
            if m.score >= RELEVANCE_THRESHOLD:
                item = {"id": m.id, "score": m.score, **m.metadata}
                items.append(item)

        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:MAX_CONTEXT_SOURCES]

    # --------------------------------------------------------
    # Context Formatting
    # --------------------------------------------------------
    def _format_context(self, items):
        formatted = []
        page_refs = []

        for idx, item in enumerate(items, 1):
            content_type = item.get('content_type', 'Unknown')
            block = [f"[Source {idx}]", f"Type: {content_type}"]
            
            # Titles/Authors based on type (simple + predictable)
            title = (
                item.get("book_title")
                or item.get("article_title")
                or item.get("podcast_title")
                or item.get("interview_title")
                or item.get("youtube_title")
                or item.get("title")
            )
            if title:
                block.append(f"Title: {title}")

            author = (
                item.get("book_author")
                or item.get("article_author")
                or item.get("author")
            )
            if author:
                block.append(f"Author: {author}")

            # Additional metadata if present
            for key in ["host", "guest", "channel_name", "publication_date",
                        "upload_date", "episode_number", "journal", "year"]:
                if key in item:
                    block.append(f"{key.replace('_',' ').title()}: {item[key]}")

            # Content body
            text = (
                item.get("full_text")
                or item.get("chunk_text")
                or item.get("text")
                or item.get("content")
            )
            if text:
                block.append("\n" + text.strip())

            # URL
            url = item.get("url") or item.get("source_url")
            if url:
                block.append(f"\nURL: {url}")

            if "page_number" in item:
                page_refs.append({
                    "book": item.get("book_title", "Unknown"),
                    "author": item.get("book_author", "Unknown"),
                    "page": item["page_number"],
                    "source_num": idx
                })

            formatted.append("\n".join(block))

        return "\n\n".join(formatted), page_refs

    # --------------------------------------------------------
    # LLM Response Generation
    # --------------------------------------------------------
    def generate_response(self, query: str, context_items, session_id: str = "default", video_analysis=None):
        """
        Generate response with optional video analysis context.
        
        Args:
            query: The athlete's question
            context_items: RAG context from knowledge base
            session_id: Session identifier
            video_analysis: Optional dict containing video analysis data (JSON from match analysis)
        """
        # Build video context if available
        video_context = ""
        if video_analysis:
            try:
                player_1_stats = video_analysis.get('player_stats', {}).get('1', {})
                player_2_stats = video_analysis.get('player_stats', {}).get('2', {})
                rallies = video_analysis.get('rallies', [])
                statistics = video_analysis.get('statistics', {})
                video_info = video_analysis.get('video_info', {})
                
                video_context = f"""

MATCH PERFORMANCE DATA:
- Match Duration: {video_info.get('duration_seconds', 0):.1f} seconds ({video_info.get('total_frames', 0)} frames)
- Total Rallies: {len(rallies)}
- Score: Player 1 = {video_analysis.get('score', {}).get('player_1', 0)}, Player 2 = {video_analysis.get('score', {}).get('player_2', 0)}

PLAYER 1 PERFORMANCE:
- Average Shot Speed: {player_1_stats.get('avg_shot_speed_kmh', 0):.1f} km/h
- Maximum Shot Speed: {player_1_stats.get('max_shot_speed_kmh', 0):.1f} km/h
- Winners: {player_1_stats.get('winners', 0)}
- Errors (Unforced + Forced): {player_1_stats.get('errors', 0)}
- Points Won: {player_1_stats.get('points_won', 0)}
- Total Shots: {player_1_stats.get('total_shots', 0)}

PLAYER 2 PERFORMANCE:
- Average Shot Speed: {player_2_stats.get('avg_shot_speed_kmh', 0):.1f} km/h
- Maximum Shot Speed: {player_2_stats.get('max_shot_speed_kmh', 0):.1f} km/h
- Winners: {player_2_stats.get('winners', 0)}
- Errors (Unforced + Forced): {player_2_stats.get('errors', 0)}
- Points Won: {player_2_stats.get('points_won', 0)}
- Total Shots: {player_2_stats.get('total_shots', 0)}

MATCH STATISTICS:
- Maximum Ball Speed: {statistics.get('max_ball_speed_kmh', 0):.1f} km/h
- Player 1 Max Speed: {statistics.get('max_player1_speed_kmh', 0):.1f} km/h
- Player 2 Max Speed: {statistics.get('max_player2_speed_kmh', 0):.1f} km/h

Use this performance data to provide specific, actionable mental coaching advice. Connect the technical performance metrics to mental game strategies.
"""
            except Exception as e:
                print(f"Warning: Could not format video analysis context: {e}")
                video_context = ""
        
        if not context_items:
            prompt = f"""
You are an experienced mental performance coach for athletes.
The athlete asked: {query}
{video_context}

Use your knowledge to provide helpful advice. Be supportive and give actionable guidance. If video analysis data is provided, use it to give specific, personalized advice.
"""
            page_refs = []
        else:
            context_text, page_refs = self._format_context(context_items)
            
            prompt = f"""
You are an experienced mental performance coach for athletes.
Use the knowledge base context below when relevant, speak supportively, and give actionable advice.

Knowledge Base Context:
{context_text}
{video_context}

Athlete question: {query}

When providing advice, reference specific performance metrics from the match data when relevant. For example, if the player made many errors, address error management. If speed decreased over time, discuss maintaining intensity and mental stamina.
"""

        try:
            # Use Claude 4 Sonnet API
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=700,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract text from Claude response
            text = self._extract_claude_text(message)
            
            # Ensure we have a string, not an object
            if not isinstance(text, str):
                text = str(text) if text is not None else "Unable to generate response."
            
            if not text or text.strip() == "":
                text = "I couldn't generate a response. Please try again."

            # Add references
            if page_refs:
                source_text = "\n\n" + "─" * 60 + "\nSources:\n"
                for ref in page_refs:
                    source_text += f"• {ref['book']} by {ref['author']} (Page {ref['page']})\n"
                text += source_text

            return text

        except Exception as e:
            import traceback
            print("LLM error:", e)
            print("Traceback:", traceback.format_exc())
            return "There was an error generating your response."

    # --------------------------------------------------------
    # Safe Text Extraction
    # --------------------------------------------------------
    def _extract_claude_text(self, message):
        """Extract text from Claude API response."""
        try:
            if hasattr(message, "content") and isinstance(message.content, list):
                parts = []
                for block in message.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                cleaned = "".join(parts).strip()
                if cleaned:
                    return cleaned
        except:
            pass

        if hasattr(message, "text") and message.text:
            return str(message.text).strip()

        return "Unable to extract response text."
