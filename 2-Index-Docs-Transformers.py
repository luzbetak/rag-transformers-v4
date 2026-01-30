#!/usr/bin/env python3

"""
2-Index-Docs-Transformers-v5.py
Index documents with Transformers v5 optimizations into MongoDB
Interactive menu with options to initialize, process, or run all
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{Config.LOGS_DIR}/indexing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Colors
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class OptimizedEmbedder:
    """Optimized embedder using config settings."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.MODEL_NAME
        
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if torch.cuda.is_available() and Config.GPU_ENABLED:
            self.model = self.model.cuda()
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling strategy."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts to embeddings."""
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch = texts[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_SEQ_LENGTH,
                    return_tensors="pt",
                )
                
                if torch.cuda.is_available() and Config.GPU_ENABLED:
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                outputs = self.model(**encoded)
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().numpy())
        
        return np.array(all_embeddings)


class RAGIndexer:
    """Index documents with MongoDB."""
    
    def __init__(self, mongo_uri: str = None):
        if mongo_uri is None:
            mongo_uri = Config.MONGODB_URI
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        self.embedder = OptimizedEmbedder()
        
        logger.info("Connected to MongoDB")
    
    def initialize_database(self) -> None:
        """Delete all documents from collection."""
        print(f"\n{Colors.YELLOW}Clearing MongoDB collection...{Colors.RESET}")
        result = self.collection.delete_many({})
        print(f"{Colors.GREEN}Deleted {result.deleted_count} documents{Colors.RESET}\n")
    
    def index_documents(self, documents: List[Dict]) -> Dict:
        """Index documents to MongoDB."""
        print(f"\n{Colors.CYAN}Indexing {len(documents)} documents...{Colors.RESET}")
        start_time = time.time()
        
        texts = [doc["text"] for doc in documents]
        print(f"{Colors.CYAN}Generating embeddings...{Colors.RESET}")
        embeddings = self.embedder.encode(texts)
        
        print(f"{Colors.CYAN}Saving to MongoDB...{Colors.RESET}")
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding.tolist()
            doc["indexed_at"] = datetime.utcnow()
            
            self.collection.update_one(
                {"_id": doc.get("_id")},
                {"$set": doc},
                upsert=True,
            )
        
        elapsed = time.time() - start_time
        
        stats = {
            "total_documents": len(documents),
            "indexed_count": len(documents),
            "total_time_seconds": elapsed,
            "documents_per_second": len(documents) / elapsed if elapsed > 0 else 0,
        }
        
        print(f"\n{Colors.GREEN}Indexing complete!{Colors.RESET}")
        print(f"{Colors.CYAN}Statistics:{Colors.RESET}")
        print(f"  • Documents indexed: {stats['indexed_count']}")
        print(f"  • Time: {stats['total_time_seconds']:.2f}s")
        print(f"  • Speed: {stats['documents_per_second']:.1f} docs/sec\n")
        
        return stats


def print_header():
    """Print application header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*50}{Colors.RESET}")
    print(f"{Colors.BOLD}Orthomolecular Medicine Indexing System{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*50}{Colors.RESET}\n")


def print_menu():
    """Print interactive menu."""
    print(f"{Colors.YELLOW}Options:{Colors.RESET}")
    print(f"  1. Initialize database (will delete existing data)")
    print(f"  2. Process chunks and create embeddings")
    print(f"  3. Run all operations and exit")
    print(f"  4. Exit")
    print()


def load_chunks() -> List[Dict]:
    """Load chunks from JSON file."""
    input_file = f"{Config.DATA_DIR}/large_text_chunks.json"
    
    if not Path(input_file).exists():
        print(f"{Colors.RED}File not found: {input_file}{Colors.RESET}")
        print(f"{Colors.YELLOW}Run 1-Chunk-Large-Text-Docs.py first{Colors.RESET}\n")
        return []
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    print(f"{Colors.GREEN}Loaded {len(chunks)} chunks{Colors.RESET}\n")
    return chunks


def main():
    """Main execution."""
    print_header()
    print(Config)
    
    indexer = RAGIndexer()
    
    while True:
        print_menu()
        choice = input(f"{Colors.CYAN}Enter your choice (1-4): {Colors.RESET}").strip()
        
        if choice == "1":
            indexer.initialize_database()
        
        elif choice == "2":
            chunks = load_chunks()
            if chunks:
                indexer.index_documents(chunks)
        
        elif choice == "3":
            print(f"\n{Colors.CYAN}Running all operations...{Colors.RESET}")
            indexer.initialize_database()
            chunks = load_chunks()
            if chunks:
                indexer.index_documents(chunks)
            print(f"{Colors.GREEN}All operations complete!{Colors.RESET}")
            print(f"{Colors.CYAN}Next: Run ./3-Search-Summarize.py to search documents{Colors.RESET}\n")
            break
        
        elif choice == "4":
            print(f"{Colors.YELLOW}Exiting...{Colors.RESET}\n")
            break
        
        else:
            print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}\n")


if __name__ == "__main__":
    main()
