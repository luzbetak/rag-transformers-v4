#!/usr/bin/env python3

"""
4-Search-Summarize.py
Perform semantic search on indexed documents
ChatGPT-style comprehensive answers with proper summarization
"""

import logging
import time
import warnings
import re
from typing import List, Dict
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*resume_download.*")

from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI Colors
class Colors:
    GRAY = '\033[90m'
    WHITE = '\033[97m'
    RESET = '\033[0m'


class OptimizedSearcher:
    """Search with optimized embeddings and AI summarization."""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.MODEL_NAME
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if torch.cuda.is_available() and Config.GPU_ENABLED:
            self.model = self.model.cuda()
        
        self.model.eval()
        
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        
        logger.info("Loading summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model=Config.SUMMARIZATION_MODEL,
            device=0 if torch.cuda.is_available() and Config.GPU_ENABLED else -1
        )
        
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_name = "CPU"
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query."""
        with torch.no_grad():
            encoded = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=Config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            
            if torch.cuda.is_available() and Config.GPU_ENABLED:
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            outputs = self.model(**encoded)
            embedding = self._mean_pooling(outputs, encoded["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy()
    
    def search(self, query: str, num_results: int = None) -> tuple:
        """Search documents."""
        if num_results is None:
            num_results = Config.TOP_K
        
        logger.info(f"Searching for: {query}")
        start = time.time()
        
        query_embedding = self.encode_query(query)
        all_docs = list(self.collection.find({}, {"embedding": 1, "text": 1}))
        
        results = []
        for doc in all_docs:
            if "embedding" not in doc:
                continue
            
            doc_embedding = np.array(doc["embedding"])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-9
            )
            
            if similarity >= Config.SIMILARITY_THRESHOLD:
                results.append({
                    "_id": str(doc["_id"]),
                    "text": doc["text"].strip(),
                    "similarity": float(similarity),
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        elapsed = time.time() - start
        
        return results[:num_results], elapsed
    
    def fix_spacing(self, text: str) -> str:
        """Fix words connected without spaces."""
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        replacements = [
            ('fightinginfection', 'fighting infection'),
            ('iseffective', 'is effective'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def generate_answer(self, query: str, results: List[Dict]) -> str:
        """Generate comprehensive answer using AI summarization."""
        if not results:
            return "I couldn't find relevant information about this topic."
        
        combined_text = " ".join([r["text"] for r in results[:5]])
        combined_text = combined_text[:2000]
        
        try:
            summary = self.summarizer(
                combined_text, 
                max_length=Config.MAX_LENGTH, 
                min_length=Config.MIN_LENGTH, 
                do_sample=False
            )
            answer = summary[0]["summary_text"]
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            answer = combined_text[:500]
        
        answer = self.fix_spacing(answer)
        return answer
    
    def interactive_search(self):
        """Interactive search interface."""
        print(f"\n{Colors.GRAY}" + "="*74)
        print(f"| Orthomolecular Medicine Search")
        print(f"| " + "="*72)
        print(f"| Using GPU: {self.gpu_name}")
        print(f"| Enter 'exit' to quit")
        print("="*74 + f"{Colors.RESET}\n")
        
        try:
            while True:
                query = input(f"{Colors.GRAY}Enter your question: {Colors.RESET}").strip()
                
                if query.lower() == "exit":
                    print(f"\n{Colors.GRAY}Goodbye!{Colors.RESET}\n")
                    break
                
                if not query:
                    continue
                
                print(f"\n{Colors.GRAY}🔍 Searching...{Colors.RESET}")
                results, search_time = self.search(query)
                
                if not results:
                    print(f"{Colors.GRAY}No results found.{Colors.RESET}\n")
                    continue
                
                print(f"{Colors.GRAY}💭 Generating answer...{Colors.RESET}")
                answer = self.generate_answer(query, results)
                
                print(f"\n{Colors.GRAY}" + "="*100)
                print(f"📚 Answer")
                print("="*100 + f"{Colors.RESET}\n")
                print(f"{Colors.WHITE}{answer}{Colors.RESET}")
                
                print(f"\n{Colors.GRAY}" + "="*100)
                print(f"✅ Response based on {len(results)} relevant sections (search: {search_time*1000:.0f}ms)")
                print("="*100 + f"{Colors.RESET}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GRAY}Goodbye!{Colors.RESET}\n")


if __name__ == "__main__":
    searcher = OptimizedSearcher()
    searcher.interactive_search()
