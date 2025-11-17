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

def extract_paper_info(file_path):
    """Extract paper title from first line, authors from second line, and optionally journal/year from content."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    
    # First line is the paper title
    title = lines[0].strip() if len(lines) > 0 else "Unknown"
    
    # Second line is the authors
    authors = lines[1].strip() if len(lines) > 1 else "Unknown"
    
    # Try to extract journal and year from the content
    # Look for common patterns like "Journal Name Year" or "Journal Name, Year" or "doi: 10.xxxx/xxxx"
    journal = "Unknown"
    year = "Unknown"
    
    # Read full content to search for journal/year
    content = ' '.join(lines)
    
    # Try to find year (4-digit year between 1900-2100)
    # Use pattern that captures full years (not just groups)
    full_years = re.findall(r'\b(19\d{2}|20\d{2})\b', content)
    if full_years:
        # Get the most recent year found (likely the publication year)
        valid_years = [int(y) for y in full_years if 1900 <= int(y) <= 2100]
        if valid_years:
            year = max(valid_years)
    
    # Try to find journal name (common patterns)
    # Look for patterns like "Journal of..." or common journal indicators
    journal_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Journal|Review|Quarterly|Magazine|Psychology|Science))',
        r'([A-Z][a-zA-Z\s]+(?:Journal|Review|Quarterly|Magazine))',
        r'doi:\s*10\.\d+/([^\s,]+)',  # Extract from DOI
    ]
    
    for pattern in journal_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            journal = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
            # Clean up journal name
            journal = journal.strip().rstrip(',.')
            # Remove common prefixes/suffixes
            journal = re.sub(r'^(Received|Accepted|Corresponding|doi:).*', '', journal, flags=re.IGNORECASE).strip()
            if len(journal) > 100 or len(journal) < 3:  # Too long or too short
                journal = "Unknown"
            else:
                break
    
    # If journal still unknown, try to extract from lines 3-8 (common location for journal info)
    if journal == "Unknown" and len(lines) > 2:
        for i in range(2, min(8, len(lines))):
            line = lines[i].strip()
            # Check if line looks like a journal name or contains journal info
            if any(keyword in line.lower() for keyword in ['journal', 'review', 'quarterly', 'magazine', 'publication', 'psychology', 'science']):
                # Extract just the journal name part (before year or other metadata)
                journal_match = re.search(r'([A-Z][^0-9]+?)(?:\s+\d{4}|\s+Vol\.|\s+doi:)', line)
                if journal_match:
                    journal = journal_match.group(1).strip().rstrip(',.')
                else:
                    journal = line[:100]  # Limit length
                if len(journal) > 3:
                    break
    
    return title, authors, journal, year

def extract_text_chunks(file_path, paper_title, authors, journal, year, chunk_size=2500, overlap=300):
    """Extract text chunks from scientific paper for embedding.
    
    Uses sentence-aware chunking to preserve context and maintain readability.
    """
    print(f"Extracting text chunks for '{paper_title}' by {authors}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        # Skip the first two lines (title and authors)
        lines = content.split('\n')
        if len(lines) > 2:
            # Skip first two lines (title and authors), keep the rest
            paper_text = '\n'.join(lines[2:])
        else:
            paper_text = content
    
    # Clean the text while preserving sentence structure
    # Normalize multiple newlines to single newlines, but keep paragraph breaks
    cleaned_content = re.sub(r'\n{3,}', '\n\n', paper_text)
    # Normalize multiple spaces to single spaces
    cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)
    # Remove spaces at start/end of lines
    cleaned_content = re.sub(r'^ +| +$', '', cleaned_content, flags=re.MULTILINE)
    
    # Split into sentences (preserve sentence boundaries)
    # Split on sentence endings: . ! ? followed by space or newline
    sentence_pattern = r'([.!?])\s+'
    sentences = re.split(sentence_pattern, cleaned_content)
    
    # Reconstruct sentences (rejoin with their punctuation)
    reconstructed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            if sentence.strip():
                reconstructed_sentences.append(sentence.strip())
        elif sentences[i].strip():
            reconstructed_sentences.append(sentences[i].strip())
    
    # If sentence splitting didn't work well, fall back to word-based chunking
    if len(reconstructed_sentences) < 3:
        # Fallback: split by words
        words = cleaned_content.split()
        chunks = []
        step_size = chunk_size - overlap
        for j in range(0, len(words), step_size):
            chunk_words = words[j:j + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'id': f"{paper_title.replace(' ', '_').replace('/', '_')}_chunk_{j//step_size}",
                    'text': chunk_text.strip(),
                    'content_type': 'scientific_paper',
                    'paper_title': paper_title,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                })
        return chunks
    
    # Sentence-based chunking
    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_idx = 0
    
    for sentence in reconstructed_sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed chunk size, save current chunk
        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'id': f"{paper_title.replace(' ', '_').replace('/', '_')}_chunk_{chunk_idx}",
                    'text': chunk_text.strip(),
                    'content_type': 'scientific_paper',
                    'paper_title': paper_title,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                })
                chunk_idx += 1
            
            # Start new chunk with overlap (keep last sentences that fit within overlap word count)
            if overlap > 0 and current_chunk:
                overlap_sentences = []
                overlap_word_count = 0
                # Add sentences from the end until we reach overlap word count
                for sentence in reversed(current_chunk):
                    sentence_words = len(sentence.split())
                    if overlap_word_count + sentence_words <= overlap:
                        overlap_sentences.insert(0, sentence)
                        overlap_word_count += sentence_words
                    else:
                        break
                current_chunk = overlap_sentences
                current_word_count = overlap_word_count
            else:
                current_chunk = []
                current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.strip()) > 100:
            chunks.append({
                'id': f"{paper_title.replace(' ', '_').replace('/', '_')}_chunk_{chunk_idx}",
                'text': chunk_text.strip(),
                'content_type': 'scientific_paper',
                'paper_title': paper_title,
                'authors': authors,
                'journal': journal,
                'year': year,
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
    """Process and upload scientific paper chunks (text + metadata) to Pinecone."""
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
                            'paper_title': item['paper_title'],
                            'authors': item['authors'],
                            'journal': item['journal'],
                            'year': item['year'],
                            'content_type': item['content_type']
                        }
                    })

            # Prepare vectors for upload
            vectors_to_upsert = []
            for idx, item in enumerate(batch_embeddings):
                values = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                vector_id = f"paper_{i + idx}_{hash(item['metadata']['paper_title']) % 10000}"
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
    """Main function to process all scientific papers in the scientific_papers directory and upload embeddings."""
    print("Starting scientific paper vectorization process...")
    print(f"Using embedding model: text-embedding-004 (dimension: {dimension})")
    print(f"Target Pinecone index: {index_name}")
    
    # Construct path to the scientific_papers directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    papers_dir = os.path.join(project_root, 'RAG_Data', 'scientfiic_papers')  # Note: keeping the typo in directory name
    
    # Also check for correctly spelled directory name
    if not os.path.exists(papers_dir):
        papers_dir_alt = os.path.join(project_root, 'RAG_Data', 'scientific_papers')
        if os.path.exists(papers_dir_alt):
            papers_dir = papers_dir_alt
        else:
            print(f"ERROR: Scientific papers directory not found at {papers_dir} or {papers_dir_alt}")
            print("Please ensure the 'scientfiic_papers' or 'scientific_papers' directory exists in the 'RAG_Data' directory at the project root.")
            return

    # Get all text files in the scientific_papers directory
    paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.txt')]
    
    if not paper_files:
        print(f"WARNING: No .txt files found in {papers_dir}")
        print("Please add scientific paper files (.txt format) to the scientific_papers directory.")
        print("Format: First line should be the paper title, second line should be the authors")
        return

    print(f"\nFound {len(paper_files)} scientific paper file(s) to process:")
    for paper_file in paper_files:
        print(f"  - {paper_file}")

    all_chunks = []

    # Process each scientific paper file
    for paper_file in paper_files:
        paper_path = os.path.join(papers_dir, paper_file)
        
        try:
            # Extract paper title, authors, journal, and year
            paper_title, authors, journal, year = extract_paper_info(paper_path)
            print(f"\nProcessing: {paper_file}")
            print(f"  Title: {paper_title}")
            print(f"  Authors: {authors}")
            print(f"  Journal: {journal}")
            print(f"  Year: {year}")
            
            # Extract text chunks
            chunks = extract_text_chunks(paper_path, paper_title, authors, journal, year)
            print(f"  Extracted {len(chunks)} text chunks")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"ERROR processing {paper_file}: {e}")
            continue

    print(f"\nTotal chunks extracted from all scientific papers: {len(all_chunks)}")

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
    print("Comprehensive scientific paper vectorization completed successfully!")

if __name__ == "__main__":
    main()

