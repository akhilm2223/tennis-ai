import os
import re
from pathlib import Path
from pypdf import PdfReader


def extract_author_title_from_filename(filename):
    """Extract author and title from PDF filename.
    
    Expected format: "Author - Title (year, publisher) - source.pdf"
    or variations like "Author_ Co-Author - Title (year, publisher) - source.pdf"
    """
    # Remove .pdf extension
    name = filename.replace('.pdf', '')
    
    # Try to extract author and title from various patterns
    # Pattern 1: "Author - Title (...)"
    match = re.match(r'^(.+?)\s*-\s*(.+?)\s*\(', name)
    if match:
        author = match.group(1).strip()
        title = match.group(2).strip()
        
        # Clean up author (handle underscores and commas)
        author = author.replace('_', ', ').strip()
        
        # Clean up title (handle underscores)
        title = title.replace('_', ' ').strip()
        
        return title, author
    
    # Pattern 2: "Author, Name - Title" or "Author Name - Title"
    match = re.match(r'^(.+?)\s*-\s*(.+)$', name)
    if match:
        author = match.group(1).strip().replace('_', ', ')
        title = match.group(2).strip().replace('_', ' ')
        # Remove trailing metadata if any
        title = re.sub(r'\s*\(.+$', '', title)
        return title, author
    
    # Fallback: use filename as title, unknown author
    return name.replace('_', ' '), "Unknown Author"


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        print(f"  Extracting text from {len(reader.pages)} pages...")
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--#{page_num}--\n{text}")
                if page_num % 50 == 0:
                    print(f"    Processed {page_num} pages...")
            except Exception as e:
                print(f"    Warning: Could not extract text from page {page_num}: {e}")
                continue
        
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"  Error reading PDF: {e}")
        return None


def get_metadata_from_pdf(pdf_path):
    """Try to extract title and author from PDF metadata."""
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        
        title = None
        author = None
        
        if metadata:
            title = metadata.get('/Title') or metadata.get('Title')
            author = metadata.get('/Author') or metadata.get('Author')
        
        return title, author
    except Exception:
        return None, None


def convert_pdf_to_text(pdf_path, output_dir):
    """Convert a PDF book to a text file with title and author as first two lines."""
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem  # filename without extension
    
    print(f"\nProcessing: {pdf_path.name}")
    
    # Try to get title and author from PDF metadata first
    title, author = get_metadata_from_pdf(pdf_path)
    
    # If metadata not available, parse from filename
    if not title or not author:
        title, author = extract_author_title_from_filename(pdf_path.name)
        print(f"  Using filename parsing - Title: {title[:60]}...")
        print(f"  Author: {author}")
    else:
        print(f"  Using PDF metadata - Title: {title}")
        print(f"  Author: {author}")
    
    # Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)
    
    if text_content is None:
        print(f"  Failed to extract text from {pdf_path.name}")
        return False
    
    # Create output text file
    output_filename = f"{pdf_name}.txt"
    output_path = Path(output_dir) / output_filename
    
    # Write text file with title and author as first two lines
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write(f"{author}\n")
        f.write(text_content)
    
    print(f"  ✓ Saved to: {output_path}")
    return True


def main():
    """Main function to convert all PDF books to text files."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    books_dir = project_root / 'books'
    output_dir = project_root / 'RAG_Data' / 'book'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if books directory exists
    if not books_dir.exists():
        print(f"ERROR: Books directory not found at {books_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(books_dir.glob('*.pdf'))
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {books_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        if convert_pdf_to_text(pdf_file, output_dir):
            success_count += 1
    
    print(f"\n=== CONVERSION COMPLETE ===")
    print(f"Successfully converted {success_count}/{len(pdf_files)} book(s)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

