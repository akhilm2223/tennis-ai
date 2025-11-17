import os
import re
import numpy as np
from tqdm import tqdm
import dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()

# Initialize Google Gemini client
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=google_api_key)
# Also set as environment variable for better compatibility
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Pinecone index details
index_name = "mental-coaching"  # Pinecone requires lowercase alphanumeric or hyphens
dimension = 768  # text-embedding-004 uses 768 dimensions
metric = "cosine"

# Check if the index exists
existing_indexes = []
for index in pc.list_indexes():
    existing_indexes.append(index.name)

# create index if it doesn't exist
if index_name not in existing_indexes:
    print(f"Index '{index_name}' does not exist. Creating index...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'  
        )
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

# Access the index
index = pc.Index(index_name)

def extract_youtube_info(file_path):
    """Extract video title from first line and youtuber/channel from second line of the text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()
    
    # First line is the video title
    title = first_line if first_line else "Unknown"
    
    # Second line is the channel/youtuber name
    youtuber = second_line if second_line else "Unknown"
    
    return title, youtuber

def extract_text_chunks(file_path, video_title, youtube_channel, chunk_size=2500, overlap=300):
    """Extract text chunks from YouTube video transcript for embedding."""
    print(f"Extracting text chunks for '{video_title}' by {youtube_channel}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        # Skip the first two lines (title and channel)
        lines = content.split('\n', 2)
        if len(lines) > 2:
            transcript_text = lines[2]
        else:
            transcript_text = content
    
    # Clean the text
    cleaned_content = re.sub(r'\n+', '\n', transcript_text)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    
    chunks = []
    words = cleaned_content.split()
    
    for j in range(0, len(words), chunk_size - overlap):
        chunk_words = words[j:j + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 100:
            chunks.append({
                'id': f"{video_title.replace(' ', '_')}_chunk_{j//(chunk_size - overlap)}",
                'text': chunk_text.strip(),
                'content_type': 'youtube_video',
                'youtube_title': video_title,
                'youtube_channel': youtube_channel,
            })
    
    return chunks

def generate_embedding(text):
    """Generate embedding for a single text using Google's text-embedding-004 (768 dimensions)."""
    try:
        # Ensure API key is configured before each call (library may reset state)
        genai.configure(api_key=google_api_key)
        
        # Use text-embedding-004 which produces 768 dimensions
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"  # For storing documents
        )
        embedding = np.array(result['embedding'])
        
        # Verify dimensions match expected
        if len(embedding) != 768:
            print(f"WARNING: Embedding dimension {len(embedding)} does not match expected 768")
        
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
        return None  # Return None on failure for a clear check

def process_and_upload_chunks(chunks, batch_size: int = 100):
    """Process and upload YouTube video chunks (text + metadata) to Pinecone."""
    try:
        print(f"\nProcessing {len(chunks)} text chunks...")
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding text chunks"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = []

            # Generate embeddings for batch
            for item in batch:
                embedding = generate_embedding(item['text'])
                if embedding is not None:
                    batch_embeddings.append({
                        'embedding': embedding,
                        'metadata': {
                            'full_text': item['text'],
                            'youtube_title': item['youtube_title'],
                            'youtube_channel': item['youtube_channel'],
                            'content_type': item['content_type']
                        }
                    })

            # Prepare vectors for upload
            vectors_to_upsert = []
            for idx, item in enumerate(batch_embeddings):
                values = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                vector_id = f"youtube_{i + idx}_{hash(item['metadata']['youtube_title']) % 10000}"
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': values,
                    'metadata': item['metadata']
                })

            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded batch of {len(vectors_to_upsert)} chunk vectors")

        print("Text chunk upload complete!")

    except Exception as e:
        print(f"Error processing and uploading text chunks: {e}")
        raise

def main():
    """Main function to process all YouTube videos in the youtube directory and upload embeddings."""
    print("Starting YouTube video vectorization process...")
    print(f"Using embedding model: text-embedding-004 (dimension: {dimension})")
    print(f"Target Pinecone index: {index_name}")
    
    # Construct path to the youtube directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    youtube_dir = os.path.join(project_root, 'RAG_Data', 'youtube')

    if not os.path.exists(youtube_dir):
        print(f"ERROR: YouTube directory not found at {youtube_dir}")
        print("Please ensure the 'youtube' directory exists in the 'RAG_Data' directory at the project root.")
        return

    # Get all text files in the youtube directory
    youtube_files = [f for f in os.listdir(youtube_dir) if f.endswith('.txt')]
    
    if not youtube_files:
        print(f"WARNING: No .txt files found in {youtube_dir}")
        print("Please add YouTube video transcript files (.txt format) to the youtube directory.")
        print("Format: First line should be the video title, second line should be the channel name")
        return

    print(f"\nFound {len(youtube_files)} YouTube video file(s) to process:")
    for youtube_file in youtube_files:
        print(f"  - {youtube_file}")

    all_chunks = []

    # Process each YouTube video file
    for youtube_file in youtube_files:
        youtube_path = os.path.join(youtube_dir, youtube_file)
        
        try:
            # Extract video title and youtuber from first line
            video_title, youtube_channel = extract_youtube_info(youtube_path)
            print(f"\nProcessing: {youtube_file}")
            print(f"  Title: {video_title}")
            print(f"  Channel: {youtube_channel}")
            
            # Extract text chunks
            chunks = extract_text_chunks(youtube_path, video_title, youtube_channel)
            print(f"  Extracted {len(chunks)} text chunks")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"ERROR processing {youtube_file}: {e}")
            continue

    print(f"\nTotal chunks extracted from all YouTube videos: {len(all_chunks)}")

    # Upload all chunks
    if all_chunks:
        print("\n=== UPLOADING TEXT CHUNKS ===")
        process_and_upload_chunks(all_chunks)
    else:
        print("\nNo chunks to upload.")

    # Check final index stats
    index_stats = index.describe_index_stats()
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total vectors in index '{index_name}': {index_stats['total_vector_count']}")
    print("Comprehensive YouTube video vectorization completed successfully!")

if __name__ == "__main__":
    main()

