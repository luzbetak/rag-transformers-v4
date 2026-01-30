#!/usr/bin/env python3

"""
config.py
Configuration settings for the Orthomolecular Medicine RAG system
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the Orthomolecular Medicine RAG system."""
    
    # Database Configuration
    MONGODB_URI          = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME        = os.getenv("DATABASE_NAME", "books")
    COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "chunks")
    
    # Model Configuration
    MODEL_NAME           = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DIMENSION     = int(os.getenv("VECTOR_DIMENSION", "384"))
    
    # Embedding Configuration
    BATCH_SIZE           = int(os.getenv("BATCH_SIZE", "4"))
    MAX_SEQ_LENGTH       = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    
    # Search Configuration
    TOP_K                = int(os.getenv("TOP_K", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    
    # Generation Configuration
    MAX_LENGTH           = int(os.getenv("MAX_LENGTH", "768"))
    MIN_LENGTH           = int(os.getenv("MIN_LENGTH", "100"))
    SUMMARIZATION_MODEL  = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
    
    # Chunk Configuration
    CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "128"))
    
    # Source Configuration
    SOURCE_DIR           = os.getenv("SOURCE_DIR", "source")
    DATA_DIR             = os.getenv("DATA_DIR", "data")
    LOGS_DIR             = os.getenv("LOGS_DIR", "logs")
    
    # System Configuration
    DEBUG                = os.getenv("DEBUG", "False").lower() == "true"
    GPU_ENABLED          = os.getenv("GPU_ENABLED", "True").lower() == "true"
    
    def __repr__(self):
        """Pretty print configuration."""
        config_str = "\n" + "="*70 + "\n"
        config_str += "📊 RAG System Configuration\n"
        config_str += "="*70 + "\n"
        
        config_dict = {
            "Database": {
                "MONGODB_URI": self.MONGODB_URI,
                "DATABASE_NAME": self.DATABASE_NAME,
                "COLLECTION_NAME": self.COLLECTION_NAME,
            },
            "Model": {
                "MODEL_NAME": self.MODEL_NAME,
                "VECTOR_DIMENSION": self.VECTOR_DIMENSION,
            },
            "Embedding": {
                "BATCH_SIZE": self.BATCH_SIZE,
                "MAX_SEQ_LENGTH": self.MAX_SEQ_LENGTH,
            },
            "Search": {
                "TOP_K": self.TOP_K,
                "SIMILARITY_THRESHOLD": self.SIMILARITY_THRESHOLD,
            },
            "Generation": {
                "MAX_LENGTH": self.MAX_LENGTH,
                "MIN_LENGTH": self.MIN_LENGTH,
                "SUMMARIZATION_MODEL": self.SUMMARIZATION_MODEL,
            },
            "Chunks": {
                "CHUNK_SIZE": self.CHUNK_SIZE,
                "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            },
            "Paths": {
                "SOURCE_DIR": self.SOURCE_DIR,
                "DATA_DIR": self.DATA_DIR,
                "LOGS_DIR": self.LOGS_DIR,
            },
            "System": {
                "DEBUG": self.DEBUG,
                "GPU_ENABLED": self.GPU_ENABLED,
            },
        }
        
        for section, settings in config_dict.items():
            config_str += f"\n{section}:\n"
            for key, value in settings.items():
                config_str += f"  {key}: {value}\n"
        
        config_str += "="*70 + "\n"
        return config_str
