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

def extract_article_info(file_path):
    """Extract article title from second line and author from third line of the text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    
    # First line is URL, second line is title, third line is author
    url = lines[0].strip() if len(lines) > 0 else "Unknown"
    title = lines[1].strip() if len(lines) > 1 else "Unknown"
    author = lines[2].strip() if len(lines) > 2 else "Unknown"
    
    # Clean up "Not available" author entries
    if author.lower() in ["not available", " not available"]:
        author = "Unknown"
    
    return title, author, url

def extract_text_chunks(file_path, article_title, article_author, article_url, chunk_size=2500, overlap=300):
    """Extract text chunks from article for embedding."""
    print(f"Extracting text chunks for '{article_title}' by {article_author}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        # Skip the first 4 lines (URL, title, author, date) and empty line (5th line)
        # Split with maxsplit=5 to get 6 elements: lines[0-4] are the header, lines[5] is the content
        lines = content.split('\n', 5)
        if len(lines) > 5:
            article_text = lines[5]  # Content starts after the 5th line
        elif len(lines) > 4:
            article_text = lines[4]  # Fallback if no content after 5th line
        else:
            article_text = '\n'.join(lines[3:]) if len(lines) > 3 else content  # Fallback to remaining content
    
    # Clean the text
    cleaned_content = re.sub(r'\n+', '\n', article_text)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    
    chunks = []
    words = cleaned_content.split()
    
    for j in range(0, len(words), chunk_size - overlap):
        chunk_words = words[j:j + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 100:
            chunks.append({
                'id': f"{article_title.replace(' ', '_')}_chunk_{j//(chunk_size - overlap)}",
                'text': chunk_text.strip(),
                'content_type': 'article',
                'article_title': article_title,
                'article_author': article_author,
                'article_url': article_url,
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
    """Process and upload article chunks (text + metadata) to Pinecone."""
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
                            'article_title': item['article_title'],
                            'article_author': item['article_author'],
                            'article_url': item['article_url'],
                            'content_type': item['content_type']
                        }
                    })

            # Prepare vectors for upload
            vectors_to_upsert = []
            for idx, item in enumerate(batch_embeddings):
                values = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                vector_id = f"article_{i + idx}_{hash(item['metadata']['article_title']) % 10000}"
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
    """Main function to process all articles in the articles directory and upload embeddings."""
    print("Starting article vectorization process...")
    print(f"Using embedding model: text-embedding-004 (dimension: {dimension})")
    print(f"Target Pinecone index: {index_name}")
    
    # Construct path to the articles directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    articles_dir = os.path.join(project_root, 'RAG_Data', 'articles')

    if not os.path.exists(articles_dir):
        print(f"ERROR: Articles directory not found at {articles_dir}")
        print("Please ensure the 'articles' directory exists in the 'RAG_Data' directory at the project root.")
        return

    # Get all text files in the articles directory
    article_files = [f for f in os.listdir(articles_dir) if f.endswith('.txt')]
    
    if not article_files:
        print(f"WARNING: No .txt files found in {articles_dir}")
        print("Please add article files (.txt format) to the articles directory.")
        print("Format: First line should be URL, second line should be title, third line should be author")
        return

    print(f"\nFound {len(article_files)} article file(s) to process:")
    for article_file in article_files:
        print(f"  - {article_file}")

    all_chunks = []

    # Process each article file
    for article_file in article_files:
        article_path = os.path.join(articles_dir, article_file)
        
        try:
            # Extract article title, author, and URL
            article_title, article_author, article_url = extract_article_info(article_path)
            print(f"\nProcessing: {article_file}")
            print(f"  Title: {article_title}")
            print(f"  Author: {article_author}")
            print(f"  URL: {article_url}")
            
            # Extract text chunks
            chunks = extract_text_chunks(article_path, article_title, article_author, article_url)
            print(f"  Extracted {len(chunks)} text chunks")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"ERROR processing {article_file}: {e}")
            continue

    print(f"\nTotal chunks extracted from all articles: {len(all_chunks)}")

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
    print("Comprehensive article vectorization completed successfully!")

if __name__ == "__main__":
    main()

