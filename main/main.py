import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_dataset(filepath):
    """Load JSON dataset from the given file."""
    with open('data/data.json', 'r') as f:
        data = json.load(f)
    return data

def compute_embeddings(data, model):
    """Compute embeddings for all queries in the dataset."""
    queries = [item['query'] for item in data]
    # compute embeddings and return as a NumPy array
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
    # Compute the embedding for the user query
    user_embedding = model.encode([user_query], convert_to_tensor=False)
    user_embedding = np.array(user_embedding)
    # Search the FAISS index (k=1 for the closest match)
    D, I = index.search(user_embedding, k=1)
    best_match = data[I[0][0]]
    return best_match['response']

def main():
    # Path to your dataset file (ensure this file is in your project directory)
    dataset_path = 'large_mental_health_dataset.json'
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
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


