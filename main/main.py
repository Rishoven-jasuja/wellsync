import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# **Step 1: Load Sentence Transformer Model**
model = SentenceTransformer('all-MiniLM-L6-v2')

# **Step 2: Load & Process JSON Data**
def load_all_json_files(folder_path):
    dataset = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        dataset.extend(data)
                except json.JSONDecodeError as e:
                    print(f"Error loading {filename}: {e}")
    return dataset

data_folder = 'data'
data = load_all_json_files(data_folder)

# **Step 3: Compute FAISS Embeddings**
def compute_embeddings(data, model):
    queries = [item['query'] for item in data]
    embeddings = model.encode(queries, convert_to_tensor=False)
    return np.array(embeddings)

embeddings = compute_embeddings(data, model)

# **Step 4: Build FAISS Index**
d = embeddings.shape[1]  
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# **Step 5: Intent Detection for Mental Wellness**
intents = {
    "greeting": "Hello, hi, hey",
    "emergency": "I feel suicidal, I want to hurt myself, I need urgent help",
    "self-care": "I need relaxation, I want meditation, suggest breathing exercises",
    "exit": "Bye, goodnight, exit, quit"
}
intent_embeddings = {key: model.encode(value) for key, value in intents.items()}

def detect_intent(user_message):
    new_embedding = model.encode(user_message)
    similarities = {key: cosine_similarity([new_embedding], [intent_embeddings[key]])[0][0] for key in intents}
    return max(similarities, key=similarities.get)

# **Step 6: Conversation Memory (Last 5 Messages)**
chat_history = []

def update_chat_history(user_message, bot_response):
    chat_history.append(("User", user_message))
    chat_history.append(("Bot", bot_response))
    chat_history[:] = chat_history[-5:]

def find_similar_message(user_message):
    if not chat_history:
        return None
    new_embedding = model.encode(user_message)
    previous_embeddings = [model.encode(msg[1]) for msg in chat_history]
    similarities = cosine_similarity([new_embedding], previous_embeddings)[0]
    max_sim_index = np.argmax(similarities)
    return chat_history[max_sim_index][1] if similarities[max_sim_index] > 0.7 else None

# **Step 7: Generate Response**
def get_response(user_query):
    user_embedding = model.encode([user_query], convert_to_tensor=False)
    user_embedding = np.array(user_embedding)
    D, I = index.search(user_embedding, k=1)
    
    best_match_index = I[0][0]
    best_match = data[best_match_index]
    best_match_score = D[0][0]

    detected_intent = detect_intent(user_query)
    intent_responses = {
        "greeting": "Hello! How are you feeling today?",
        "emergency": "I'm really sorry you're feeling this way. Please reach out to a friend, family member, or helpline.",
        "self-care": "Taking care of yourself is important. Would you like some meditation or breathing exercises?",
        "exit": "Okay, take care! Remember, you are not alone."
    }

    # **Check Similar Message in Chat History**
    similar_message = find_similar_message(user_query)
    if similar_message:
        return f"We discussed this earlier: {similar_message}"

    # **Check FAISS Response**
    if best_match_score < 5.0:
        return best_match['response']

    # **Check Intent-Based Response**
    if detected_intent in intent_responses:
        return intent_responses[detected_intent]

    return "I understand. Can you tell me more about how you're feeling?"

# **Step 8: Run Chatbot**
def chatbot():
    print("Mental Wellness Chatbot is Ready! Type your message (or type 'exit' to quit).")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye"]:
            print("Bot: Okay, take care! Remember, you are not alone.")
            break

        response = get_response(user_input)
        print("Chatbot:", response)
        update_chat_history(user_input, response)

if __name__ == '__main__':
    chatbot()
