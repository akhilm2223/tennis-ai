import os
import dotenv

from pinecone import Pinecone
import anthropic
import google.generativeai as genai

# Flask endpoints moved to app.py

# Load environment variables
dotenv.load_dotenv()

# Application constants
INDEX_NAME = "catalog-text-embedding-004"  # Update to your mental coaching index name
BASE_URL = "https://catalogs.rutgers.edu/generated/nb-ug_2224/"  # Update to your content base URL if needed
#default top k is the number of results to return from the search
TOP_K = 10
#default relevance threshold is the minimum score for a result to be considered relevant
RELEVANCE_THRESHOLD = 0.6
#default max context sources is the maximum number of sources to include in the context
MAX_CONTEXT_SOURCES = 10

# Initialize Anthropic Claude client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Initialize Google embeddings (for embedding generation)
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Access the knowledge base index
knowledge_index = pinecone_client.Index(INDEX_NAME)

class MentalCoachChatbot:
    """RAG chatbot for mental coaching of athletes via Pinecone + Claude."""

    def __init__(self):
        """Initialize the underlying LLM model."""
        self.model = "claude-3-5-haiku-20241022"  # Claude 4.5 Haiku
        self.client = claude_client

    def generate_query_embedding(self, query):
        """Generate an embedding vector for the user query.
        Returns None if the embedding service fails.
        """
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query",
            )
            return result['embedding']
        except Exception:
            return None

    def _query_index(self, vector, top_k):
        """Query Pinecone index and return raw matches with metadata."""
        search_results = knowledge_index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        return search_results.get('matches', [])

    def _build_content_items(self, matches, threshold=RELEVANCE_THRESHOLD):
        """Build content items from Pinecone matches, filtering by relevance threshold.
        
        Args:
            matches: List of match objects from Pinecone
            threshold: Minimum similarity score to include a match
            
        Returns:
            List of content items (dictionaries with metadata and score)
        """
        content_items = []
        
        for match in matches:
            # Filter by relevance threshold
            if match.score and match.score >= threshold:
                item = {
                    'score': match.score,
                    'id': match.id,
                    **match.metadata  # Include all metadata fields
                }
                content_items.append(item)
        
        # Sort by score descending and limit to MAX_CONTEXT_SOURCES
        content_items.sort(key=lambda x: x['score'], reverse=True)
        return content_items[:MAX_CONTEXT_SOURCES]

    def search_pinecone(self, query, top_k = TOP_K):
        """Search the Pinecone index for mental coaching content relevant to the athlete's query."""
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            return []

        try:
            # First pass
            matches = self._query_index(query_embedding, top_k=top_k)
            relevant_content = self._build_content_items(matches, threshold=RELEVANCE_THRESHOLD)
            
            if not relevant_content:
                return []

            return relevant_content

        except Exception:
            return []

    def generate_response(self, query, context_items):
        """Generate a response using Claude with provided context from Pinecone.
        
        Args:
            query: The user's query/question
            context_items: List of content items from Pinecone search (each with metadata and score)
            
        Returns:
            Generated response string, or None if generation fails
        """
        if not context_items:
            return "I couldn't find relevant mental coaching information to help with your question. Please try rephrasing your query or ask about a different aspect of mental performance."
        
        try:
            # Format context from content items
            context_text = self._format_context(context_items)
            
            # Create system message for the mental coach chatbot
            system_message = """You are an experienced mental performance coach helping athletes develop their mental game. 
You provide guidance on mental toughness, confidence, focus, motivation, handling pressure, visualization, goal setting, and overcoming challenges.

Use the provided context from mental coaching resources to help athletes. The context may come from various sources including:
- Books (with titles, authors, and page numbers)
- Articles (with titles, authors, and publication dates)
- Podcasts (with titles, hosts, guests, and episode numbers)
- Interviews (with interviewees, interviewers, and dates)
- Scientific papers (with titles, authors, journals, and years)

When referencing information, acknowledge the source type and key details (e.g., "According to [Book Title] by [Author]" or "In an interview with [Interviewee]"). 
Be supportive, empathetic, and encouraging. Provide practical, actionable advice. If the context doesn't contain enough information, 
acknowledge it but offer general guidance when appropriate. Always prioritize the athlete's mental well-being and growth."""
            
            # Create user message with query and context
            user_message = f"""Mental coaching resources:
{context_text}

Athlete's question: {query}

Please provide a helpful, supportive response based on the mental coaching context above."""
            
            # Generate response using Claude
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                return message.content[0].text
            return None
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def _format_context(self, context_items):
        """Format context items into a readable text string.
        
        Args:
            context_items: List of content items with metadata
            
        Returns:
            Formatted context string
        """
        formatted_sections = []
        
        for i, item in enumerate(context_items, 1):
            section_parts = [f"[Source {i}]"]
            
            # Determine content type and format accordingly
            content_type = item.get('content_type', 'unknown').lower()
            
            # Add content type header
            if content_type:
                content_type_display = content_type.replace('_', ' ').title()
                section_parts.insert(1, f"Type: {content_type_display}")
            
            # Handle book-specific metadata
            if content_type in ['book_text', 'book'] or 'book_title' in item:
                if 'book_title' in item:
                    section_parts.insert(1, f"Book: {item['book_title']}")
                if 'book_author' in item:
                    section_parts.insert(2, f"Author: {item['book_author']}")
                if 'page_number' in item:
                    section_parts.append(f"Page: {item['page_number']}")
            # Handle article-specific metadata
            elif content_type in ['article', 'article_text']:
                if 'article_title' in item:
                    section_parts.insert(1, f"Article: {item['article_title']}")
                if 'article_author' in item:
                    section_parts.insert(2, f"Author: {item['article_author']}")
                if 'publication_date' in item:
                    section_parts.append(f"Published: {item['publication_date']}")
            # Handle podcast-specific metadata
            elif content_type in ['podcast', 'podcast_transcript']:
                if 'podcast_title' in item:
                    section_parts.insert(1, f"Podcast: {item['podcast_title']}")
                if 'podcast_host' in item:
                    section_parts.insert(2, f"Host: {item['podcast_host']}")
                if 'episode_number' in item:
                    section_parts.append(f"Episode: {item['episode_number']}")
                if 'guest' in item:
                    section_parts.append(f"Guest: {item['guest']}")
            # Handle interview-specific metadata
            elif content_type in ['interview', 'interview_transcript']:
                if 'interview_title' in item:
                    section_parts.insert(1, f"Interview: {item['interview_title']}")
                if 'interviewee' in item:
                    section_parts.insert(2, f"Interviewee: {item['interviewee']}")
                if 'interviewer' in item:
                    section_parts.append(f"Interviewer: {item['interviewer']}")
                if 'interview_date' in item:
                    section_parts.append(f"Date: {item['interview_date']}")
            # Handle scientific paper-specific metadata
            elif content_type in ['scientific_paper', 'paper', 'research_paper']:
                if 'paper_title' in item:
                    section_parts.insert(1, f"Paper: {item['paper_title']}")
                if 'authors' in item:
                    section_parts.insert(2, f"Authors: {item['authors']}")
                if 'journal' in item:
                    section_parts.append(f"Journal: {item['journal']}")
                if 'year' in item:
                    section_parts.append(f"Year: {item['year']}")
            
            # Add generic metadata fields (fallback for content without specific type handling)
            if 'title' in item and 'book_title' not in item and 'article_title' not in item and 'podcast_title' not in item and 'interview_title' not in item and 'paper_title' not in item:
                section_parts.insert(1, f"Title: {item['title']}")
            if 'topic' in item:
                section_parts.insert(2, f"Topic: {item['topic']}")
            if 'author' in item and 'book_author' not in item and 'article_author' not in item:
                section_parts.insert(2, f"Author: {item['author']}")
            if 'category' in item:
                section_parts.insert(2, f"Category: {item['category']}")
            
            # Add text/content if available
            if 'full_text' in item:
                section_parts.append(f"\n{item['full_text']}")
            elif 'text' in item:
                section_parts.append(f"\n{item['text']}")
            elif 'content' in item:
                section_parts.append(f"\n{item['content']}")
            elif 'chunk_text' in item:
                section_parts.append(f"\n{item['chunk_text']}")

            # Add URL if available
            if 'url' in item:
                section_parts.append(f"\nURL: {item['url']}")
            elif 'source_url' in item:
                section_parts.append(f"\nURL: {item['source_url']}")
            
            formatted_sections.append("\n".join(section_parts))
        
        return "\n\n".join(formatted_sections)
