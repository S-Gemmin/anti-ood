import os
import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = 'data'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_descriptions(filepath):
    descriptions = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            descriptions.append(row['description'].strip())
    return descriptions


def main():
    print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    safe_descriptions = load_descriptions(f'{DATA_DIR}/safe.csv')
    print(f"Loaded {len(safe_descriptions)} safe descriptions")
    
    print(f"Embedding {len(safe_descriptions)} safe descriptions...")
    safe_embeddings = embedder.encode(safe_descriptions, convert_to_numpy=True)
    
    safe_data = {
        "descriptions": safe_descriptions,
        "embeddings": safe_embeddings.tolist()
    }
    
    safe_path = os.path.join(OUTPUT_DIR, "safe_embeddings.json")
    with open(safe_path, 'w') as f:
        json.dump(safe_data, f)
    print(f"Saved {len(safe_embeddings)} safe embeddings to {safe_path}")
    
    failure_descriptions = load_descriptions(f'{DATA_DIR}/failure.csv')
    print(f"Loaded {len(failure_descriptions)} failure descriptions")
    
    print(f"Embedding {len(failure_descriptions)} failure descriptions...")
    failure_embeddings = embedder.encode(failure_descriptions, convert_to_numpy=True)
    mean_failure = np.mean(failure_embeddings, axis=0)
    
    failure_data = {
        "descriptions": failure_descriptions,
        "embeddings": failure_embeddings.tolist(),
        "mean_embedding": mean_failure.tolist()
    }
    
    failure_path = os.path.join(OUTPUT_DIR, "failure_embedding.json")
    with open(failure_path, 'w') as f:
        json.dump(failure_data, f, indent=2)
    print(f"Saved failure embeddings to {failure_path}")
    
    print("Done!")


if __name__ == '__main__':
    main()