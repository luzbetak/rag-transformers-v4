#!/usr/bin/env python3
import json
from pathlib import Path

input_file = "source/The-Gerson-Therapy-Reduced.txt"
output_file = "data/input_documents.json"

print(f"Converting {input_file} to JSON...")

if not Path(input_file).exists():
    print(f"Error: {input_file} not found")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    text_content = f.read()

documents = [{
    "id": 1,
    "text": text_content,
    "source": Path(input_file).stem,
    "filename": Path(input_file).name
}]

Path(output_file).parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Created: {output_file}")
print(f"Size: {len(text_content):,} characters")
