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
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Pinecone index details
index_name = "catalog-text-embedding-004"
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

def extract_book_info(file_path):
    """Extract book title and author from the first line of the text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        first_line = file.readline().strip()
    
    # Try to parse title and author from first line
    # Format could be "Title - Author" or "Title: Author" or just "Title" etc.
    if ' - ' in first_line:
        parts = first_line.split(' - ', 1)
        title = parts[0].strip()
        author = parts[1].strip() if len(parts) > 1 else "Unknown"
    elif ': ' in first_line:
        parts = first_line.split(': ', 1)
        title = parts[0].strip()
        author = parts[1].strip() if len(parts) > 1 else "Unknown"
    else:
        title = first_line
        author = "Unknown"
    
    return title, author

def extract_page_content_with_numbers(file_path):
    """Extract page content with their actual page numbers."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Split content by page markers, capturing the page numbers
    parts = re.split(r'--- Page (\d+) ---', content)
    
    page_contents = []
    
    # Handle content before first page marker (if any)
    if parts[0].strip():
        page_contents.append((1, parts[0]))
    
    # Process pairs of (page_number, content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_number = int(parts[i])
            page_content = parts[i + 1]
            page_contents.append((page_number, page_content))
    
    return page_contents

def extract_text_chunks(file_path, book_title, book_author, chunk_size=2500, overlap=300):
    """Extract text chunks from book file for general content embedding with actual page numbers."""
    print(f"Extracting text chunks for '{book_title}' by {book_author}...")
    
    # Get page contents with their actual page numbers
    page_contents = extract_page_content_with_numbers(file_path)
    
    # If no page markers, treat entire file as one page
    if not page_contents:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Skip the first line (title and author)
            lines = content.split('\n', 1)
            if len(lines) > 1:
                page_contents = [(1, lines[1])]
            else:
                page_contents = [(1, content)]
    
    chunks = []

    for page_number, page_content in page_contents:
        cleaned_content = re.sub(r'\n+', '\n', page_content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        words = cleaned_content.split()
    
        for j in range(0, len(words), chunk_size - overlap):
            chunk_words = words[j:j + chunk_size]
            chunk_text = ' '.join(chunk_words)
        
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'id': f"{book_title.replace(' ', '_')}_chunk_{page_number}_{j//(chunk_size - overlap)}",
                    'text': chunk_text.strip(),
                    'content_type': 'book_text',
                    'page_number': page_number,
                    'book_title': book_title,
                    'book_author': book_author
                })
    
    return chunks

def generate_embedding(text):
    """Generate embedding for a single text using Google's text-embedding-004."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"  # For storing documents
        )
        return np.array(result['embedding'])
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
        return None  # Return None on failure for a clear check

def process_and_upload_chunks(chunks, batch_size: int = 100):
    """Process and upload plain text chunks (text + page_number + book metadata) to Pinecone."""
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
                            'page_number': item['page_number'],
                            'book_title': item['book_title'],
                            'book_author': item['book_author'],
                            'content_type': item['content_type']
                        }
                    })

            # Prepare vectors for upload
            vectors_to_upsert = []
            for idx, item in enumerate(batch_embeddings):
                values = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                vector_id = f"book_{i + idx}_{hash(item['metadata']['book_title']) % 10000}"
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
    """Main function to process all books in the book directory and upload embeddings."""
    print("Starting book vectorization process...")
    print(f"Using embedding model: text-embedding-004 (dimension: {dimension})")
    print(f"Target Pinecone index: {index_name}")
    
    # Construct path to the book directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    book_dir = os.path.join(project_root, 'data', 'book')

    if not os.path.exists(book_dir):
        print(f"ERROR: Book directory not found at {book_dir}")
        print("Please ensure the 'book' directory exists in the 'data' directory at the project root.")
        return

    # Get all text files in the book directory
    book_files = [f for f in os.listdir(book_dir) if f.endswith('.txt')]
    
    if not book_files:
        print(f"WARNING: No .txt files found in {book_dir}")
        print("Please add book files (.txt format) to the book directory.")
        return

    print(f"\nFound {len(book_files)} book file(s) to process:")
    for book_file in book_files:
        print(f"  - {book_file}")

    all_chunks = []

    # Process each book file
    for book_file in book_files:
        book_path = os.path.join(book_dir, book_file)
        
        try:
            # Extract book title and author from first line
            book_title, book_author = extract_book_info(book_path)
            print(f"\nProcessing: {book_file}")
            print(f"  Title: {book_title}")
            print(f"  Author: {book_author}")
            
            # Extract text chunks with page numbers
            chunks = extract_text_chunks(book_path, book_title, book_author)
            print(f"  Extracted {len(chunks)} text chunks")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"ERROR processing {book_file}: {e}")
            continue

    print(f"\nTotal chunks extracted from all books: {len(all_chunks)}")

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
    print("Comprehensive book vectorization completed successfully!")

if __name__ == "__main__":
    main()

