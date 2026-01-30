#!/usr/bin/env python3

from prettytable import PrettyTable
import numpy as np
from loguru import logger
from config import Config
from pymongo import MongoClient

# Display Configuration
DISPLAY_CONFIG = {
    'content_length': 700,     # Maximum length for content text
    'num_samples':    3,       # Number of random samples to show
    'num_embeddings': 15,      # Number of embedding values to display
    'table_width':    150,     # Width of the value column
    'field_width':    12,      # Width of the field column
}

def format_embedding(embedding, num_values=DISPLAY_CONFIG['num_embeddings']):
    """Format embedding vector preview"""
    if isinstance(embedding, (list, np.ndarray)):
        values = [f"{x:.4f}" for x in embedding[:num_values]]
        return f"[{', '.join(values)}... ({len(embedding)} dims)]"
    return str(embedding)

def truncate_text(text, max_length=DISPLAY_CONFIG['content_length']):
    """Truncate text with ellipsis"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def format_value(value):
    """Format display values with type-specific handling"""
    if isinstance(value, (list, np.ndarray)):
        return format_embedding(value)
    elif isinstance(value, dict):
        return f"<dict: {str(value)[:DISPLAY_CONFIG['content_length']]}...>"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return truncate_text(value)
    elif value is None:
        return "None"
    else:
        return f"<{type(value).__name__}>"

def explore_database():
    """Display random chunks from the database"""
    try:
        client = MongoClient(Config.MONGODB_URI)
        db = client[Config.DATABASE_NAME]
        collection = db[Config.COLLECTION_NAME]
        
        print("\n✅ Connected to MongoDB")
        total_chunks = collection.count_documents({})
        print(f"📚 Total chunks: {total_chunks}")

        # Get random samples with all fields
        pipeline = [
            {"$sample": {"size": DISPLAY_CONFIG['num_samples']}},
            {"$project": {
                "chunk_id": 1,
                "content": 1,
                "text": 1,
                "start_char": 1,
                "end_char": 1,
                "embedding": 1
            }}
        ]
        
        samples = list(collection.aggregate(pipeline))

        if not samples:
            print("\nNo chunks found")
            return

        # Display chunks
        table = PrettyTable()
        table.field_names = ["Field", "Value"]
        table.align = 'l'
        table._max_width = {
            "Field": DISPLAY_CONFIG['field_width'], 
            "Value": DISPLAY_CONFIG['table_width']
        }

        for i, sample in enumerate(samples, 1):
            print(f"\n🔍 Chunk #{i}:")
            table.clear_rows()

            # Calculate length from content or text
            if 'content' in sample:
                sample['length'] = len(sample['content'])
            elif 'text' in sample:
                sample['length'] = len(sample['text'])

            # Add span information
            if 'start_char' in sample and 'end_char' in sample:
                sample['span'] = f"{sample['start_char']} -> {sample['end_char']}"

            fields = ['chunk_id', 'content', 'text', 'length', 'span', 'embedding']
            for field in fields:
                if field in sample:
                    try:
                        formatted_value = format_value(sample[field])
                        table.add_row([field, formatted_value])
                    except Exception as e:
                        logger.error(f"Error formatting {field}: {e}")
                        table.add_row([field, f"<Error: {str(e)}>"])

            print(table)
            # print("-" * DISPLAY_CONFIG['table_width'])

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {str(e)}")

def main():
    print("\nOrthomolecular Medicine Database Explorer")
    print("=" * 50)
    explore_database()
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
