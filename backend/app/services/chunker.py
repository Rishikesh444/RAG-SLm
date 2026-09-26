import uuid
from typing import List, Dict, Any
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from app.core.config import settings

def chunk_text(
    text: str, 
    source_name: str, 
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> List[Dict[str, Any]]:
    """
    Split continuous document text into overlapping chunks.
    
    Args:
        text: Extracted raw text from document.
        source_name: Original file name (used as source metadata).
        chunk_size: Target character count per chunk.
        chunk_overlap: Overlap character count between consecutive chunks.
        
    Returns:
        List of dictionaries containing chunk details:
        [
            {
                "id": "unique-uuid",
                "text": "chunk text content...",
                "metadata": {
                    "source": "filename.pdf",
                    "chunk_index": 0,
                    "character_count": 482
                }
            }
        ]
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    # RecursiveCharacterTextSplitter tries splitting on ["\n\n", "\n", " ", ""]
    # preserving semantic paragraphs and sentences as much as possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    raw_chunks = splitter.split_text(text)
    
    processed_chunks = []
    for idx, chunk_content in enumerate(raw_chunks):
        chunk_id = f"{source_name}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_content,
            "metadata": {
                "source": source_name,
                "chunk_index": idx,
                "character_count": len(chunk_content)
            }
        })
        
    return processed_chunks
