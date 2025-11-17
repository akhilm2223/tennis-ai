import os
import re
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()

class PodcastEmbeddingGenerator:
    def __init__(self):
        """Initialize the embedding generator with Google Gemini and Pinecone clients."""
        # Initialize Google Gemini client
        google_api_key = os.getenv('GEMINI_API_KEY')
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "mental-coaching"  # Use same index as other content types
        
        # Set max characters for text chunking (Google's limit is around 20k tokens)
        self.max_chars = 15000  # Safe limit for embeddings
        self.dimension = 768  # text-embedding-004 uses 768 dimensions
        
    def setup_pinecone_index(self):
        """Create or connect to Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = []
            for index in self.pc.list_indexes():
                existing_indexes.append(index.name)
            
            if self.index_name not in existing_indexes:
                print(f"Index '{self.index_name}' does not exist. Creating index...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"Index '{self.index_name}' created successfully!")
            else:
                print(f"Index '{self.index_name}' already exists.")
            
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            raise
    
    def extract_podcast_metadata(self, filepath: str) -> Tuple[str, str]:
        """Extract podcast title from first line and host/author from second line of the text file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            first_line = file.readline().strip()
            second_line = file.readline().strip()
        
        # First line is the podcast title
        title = first_line if first_line else "Unknown"
        
        # Second line is the host/author name
        author = second_line if second_line else "Unknown"
        
        return title, author    

    def chunk_text(self, text: str, max_chars: int = None) -> List[str]:
        """Split text into chunks that fit within character limits."""
        if max_chars is None:
            max_chars = self.max_chars
            
        # If text is within limit, return as single chunk
        if len(text) <= max_chars:
            return [text]
        
        # Split into chunks by characters, trying to break at sentence boundaries
        chunks = []
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars]
            
            # Try to break at sentence boundary if not the last chunk
            if i + max_chars < len(text):
                # Look for the last sentence ending in the chunk
                last_period = chunk.rfind('.')
                last_exclamation = chunk.rfind('!')
                last_question = chunk.rfind('?')
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > max_chars * 0.7:  # Only if we find a sentence end in the last 30%
                    chunk = chunk[:sentence_end + 1]
            
            chunks.append(chunk)
        
        return chunks
    
    def generate_embedding(self, text: str):
        """Generate embedding for a text chunk using Google's text-embedding-004 (768 dimensions)."""
        try:
            # Ensure API key is configured before each call (library may reset state)
            genai.configure(api_key=self.google_api_key)
            
            # Use text-embedding-004 which produces 768 dimensions
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"  # For storing documents
            )
            embedding = np.array(result['embedding'])
            
            # Verify dimensions match expected
            if len(embedding) != self.dimension:
                print(f"WARNING: Embedding dimension {len(embedding)} does not match expected {self.dimension}")
            
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
            return None  # Return None on failure for a clear check
    
    def process_podcast_file(self, filepath: str) -> List[Dict]:
        """Process a single podcast file and return embedding data."""
        try:
            filename = os.path.basename(filepath)
            
            # Extract title and author from first two lines
            title, author = self.extract_podcast_metadata(filepath)
            
            print(f"Processing: {title} by {author}")
            
            # Read content, skipping the first two lines (title and author)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.read().split('\n', 2)
                if len(lines) > 2:
                    content = lines[2]  # Skip first two lines
                else:
                    content = ''
            
            if not content.strip():
                print(f"Skipping empty file: {filepath}")
                return []
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            embeddings_data = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    continue
                
                # Create metadata
                metadata = {
                    'full_text': chunk,
                    'podcast_title': title,
                    'podcast_host': author,
                    'content_type': 'podcast'
                }
                
                # Create unique ID
                vector_id = f"podcast_{i}_{hash(title) % 10000}"
                
                embeddings_data.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            return embeddings_data
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return []
    
    def upload_to_pinecone(self, embeddings_data: List[Dict], batch_size: int = 100):
        """Upload embeddings to Pinecone in batches."""
        try:
            total_vectors = len(embeddings_data)
            print(f"Uploading {total_vectors} vectors to Pinecone...")
            
            # Convert embeddings to lists if they're numpy arrays
            for item in embeddings_data:
                if isinstance(item['values'], np.ndarray):
                    item['values'] = item['values'].tolist()
            
            for i in tqdm(range(0, total_vectors, batch_size), desc="Uploading vectors"):
                batch = embeddings_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"Uploaded batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
            
            print("Upload completed successfully!")
            
        except Exception as e:
            print(f"Error uploading to Pinecone: {e}")
            raise
    
    def process_all_podcasts(self, podcast_dir: str = None):
        """Process all podcast files and upload embeddings to Pinecone."""
        # Construct path to the podcast directory
        if podcast_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            podcast_dir = os.path.join(project_root, 'RAG_Data', 'podcast')
        
        if not os.path.exists(podcast_dir):
            print(f"ERROR: Podcast directory not found at {podcast_dir}")
            print("Please ensure the 'podcast' directory exists in the 'RAG_Data' directory at the project root.")
            return
        
        # Setup Pinecone index
        self.setup_pinecone_index()
        
        # Get all podcast files
        podcast_files = [f for f in os.listdir(podcast_dir) if f.endswith('.txt')]
        
        if not podcast_files:
            print(f"WARNING: No .txt files found in {podcast_dir}")
            print("Please add podcast transcript files (.txt format) to the podcast directory.")
            return
        
        print(f"\nFound {len(podcast_files)} podcast file(s) to process:")
        for podcast_file in podcast_files:
            print(f"  - {podcast_file}")
        
        all_embeddings = []
        
        # Process each podcast file
        for filename in podcast_files:
            filepath = os.path.join(podcast_dir, filename)
            try:
                embeddings_data = self.process_podcast_file(filepath)
                all_embeddings.extend(embeddings_data)
            except Exception as e:
                print(f"ERROR processing {filename}: {e}")
                continue
        
        print(f"\nTotal chunks extracted from all podcasts: {len(all_embeddings)}")
        
        if all_embeddings:
            # Upload to Pinecone
            print("\n=== UPLOADING TEXT CHUNKS ===")
            self.upload_to_pinecone(all_embeddings)
            
            # Check final index stats
            index_stats = self.index.describe_index_stats()
            print(f"\n=== FINAL RESULTS ===")
            print(f"- Processed {len(podcast_files)} podcast files")
            print(f"- Generated {len(all_embeddings)} embedding vectors")
            print(f"- Total vectors in index '{self.index_name}': {index_stats['total_vector_count']}")
            print("Comprehensive podcast vectorization completed successfully!")
        else:
            print("No embeddings generated")
    
    def query_similar_podcasts(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query for similar podcast content."""
        try:
            # Ensure index is set up
            if not hasattr(self, 'index'):
                self.setup_pinecone_index()
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results.matches
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []


def main():
    """Main function to run the embedding generation process."""
    try:
        print("Starting podcast vectorization process...")
        
        # Initialize and run the embedding generator
        generator = PodcastEmbeddingGenerator()
        print(f"Using embedding model: text-embedding-004 (dimension: {generator.dimension})")
        print(f"Target Pinecone index: {generator.index_name}")
        
        generator.process_all_podcasts()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()