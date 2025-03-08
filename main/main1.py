import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_all_json_files(folder_path):
    """Load and merge all JSON files from the specified folder."""
    dataset = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        dataset.extend(data)
                    else:
                        print(f"Warning: {filename} does not contain a list of entries.")
                except json.JSONDecodeError as e:
                    print(f"Error loading {filename}: {e}")
    return dataset

def compute_embeddings(data, model):
    """Compute embeddings for all queries in the dataset."""
    queries = [item['query'] for item in data]
    embeddings = model.encode(queries, convert_to_tensor=False)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    """Build and return a FAISS index from the embeddings."""
    d = embeddings.shape[1]  # dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def get_response(user_query, model, index, data):
    """Return the response corresponding to the nearest query from the dataset."""
    user_embedding = model.encode([user_query], convert_to_tensor=False)
    user_embedding = np.array(user_embedding)
    D, I = index.search(user_embedding, k=1)
    best_match = data[I[0][0]]
    return best_match['response']

def main():
    # The data folder name is "data" (your JSON files should be placed here)
    data_folder = 'data'
    print("Loading dataset from folder:", data_folder)
    data = load_all_json_files(data_folder)
    print(f"Loaded {len(data)} entries from JSON files.")

    # Load the pre-trained SentenceTransformer model
    print("Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings for all dataset queries
    print("Computing embeddings for the dataset...")
    embeddings = compute_embeddings(data, model)
    
    # Build FAISS index for fast similarity search
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    print("\nChatbot is ready! Type your message (or type 'exit' to quit).")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_response(user_input, model, index, data)
        print("Chatbot:", response)

if __name__ == '__main__':
    main()
