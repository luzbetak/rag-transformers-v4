#!/usr/bin/env python3

"""
1-Chunk-Large-Text-Docs.py 
Chunk large text documents directly into smaller pieces
Reads directly from source/*.txt files
"""

import json
from pathlib import Path
from typing import List, Dict
import logging

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks."""
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def process_text_file(input_file: str, output_file: str, chunk_size: int = None) -> None:
    """Process text file directly and save chunks."""
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    
    logger.info(f"Loading text from {input_file}")
    
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    logger.info(f"Loaded {len(text_content):,} characters")
    
    doc_chunks = chunk_text(text_content, chunk_size=chunk_size)
    
    chunks = []
    for chunk_id, chunk_text_content in enumerate(doc_chunks):
        chunks.append({
            "_id": chunk_id,
            "chunk_id": chunk_id,
            "doc_id": 0,
            "text": chunk_text_content,
            "source": Path(input_file).stem,
        })
    
    logger.info(f"Created {len(chunks)} chunks")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "total_chunks": len(chunks),
        "total_documents": 1,
        "source_file": input_file,
        "chunks": chunks
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved chunks to {output_file}")


if __name__ == "__main__":
    print(Config)
    
    source_files = list(Path(Config.SOURCE_DIR).glob("*.txt"))
    
    if not source_files:
        logger.error(f"No .txt files found in {Config.SOURCE_DIR}")
        exit(1)
    
    input_file = str(source_files[0])
    output_file = f"{Config.DATA_DIR}/large_text_chunks.json"
    
    logger.info("Starting document chunking process...")
    process_text_file(input_file, output_file, Config.CHUNK_SIZE)
    logger.info("Document chunking complete!")
    logger.info(f"Next: Run ./2-Index-Docs-Transformers-v5.py to index the chunks")
