import os   #working
import faiss
import json
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to load dataset from multiple JSON files
def load_dataset(data_folder="data"):
    dataset = []
    intents = {}  # Dictionary to store intents
    
    if not os.path.exists(data_folder):
        print(f"Warning: Folder '{data_folder}' not found. Make sure it exists.")
        return dataset, intents

    for file in os.listdir(data_folder):
        if file.endswith(".json"):
            file_path = os.path.join(data_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    entries = json.load(f)
                    dataset.extend(entries)  # Merge all questions into one list
                    
                    # Store intent for each question (fixing the error)
                    for entry in entries:
                        intents[entry["question"]] = entry.get("intent", "unknown")  # Assign 'unknown' if missing
                except json.JSONDecodeError:
                    print(f"Error: Could not parse {file_path}. Skipping.")
    
    return dataset, intents

# Load dataset and intents
data, intents = load_dataset()

# Create FAISS index
dimension = 384  # Embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

# Encode questions & add to FAISS
question_embeddings = []
questions = []  # Store questions separately for indexing

for entry in data:
    embedding = model.encode([entry["question"]])[0]  # Get embedding
    question_embeddings.append(embedding)
    questions.append(entry["question"])  # Store original question

question_embeddings = np.array(question_embeddings).astype("float32")
index.add(question_embeddings)  # Add all embeddings to FAISS

# Conversation memory (stores last 5 messages)
conversation_memory = []
last_topic = None  # Track the last recognized topic

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

# Retrieve response from FAISS
def get_response(user_input):
    query_embedding = model.encode([user_input]).astype("float32")
    _, top_match = index.search(query_embedding, 1)  # Get best match index
    match_index = top_match[0][0]

    if match_index != -1:
        response_list = data[match_index]["response"]
        return random.choice(response_list), questions[match_index]  # Return response & matched question
    else:
        return None, None  # No match found

# Infer context from conversation memory
def infer_context(user_input):
    global last_topic
    if last_topic:
        user_input = last_topic + " " + user_input  # Append last topic to new query
    return user_input

# Chatbot function
def chatbot():
    global last_topic
    print("Chatbot: Hi! How can I help you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Chatbot: Take care! Have a great day.")
            break

        # Preprocess user input
        user_input = preprocess(user_input)

        # Check if input is vague
        if len(user_input.split()) < 4:  # Example: "what can be its cause?"
            user_input = infer_context(user_input)  # Add context

        # Retrieve response
        response, matched_question = get_response(user_input)

        # If FAISS couldn't find a match, use fallback response
        if response is None:
            response = "I'm not sure. Can you clarify what you're asking about?"
        else:
            last_topic = matched_question  # Store matched question as last topic

        # Store conversation memory (limit to last 5 interactions)
        conversation_memory.append({"user": user_input, "bot": response})
        if len(conversation_memory) > 5:
            conversation_memory.pop(0)

        print(f"Chatbot: {response}")

# Run chatbot
if __name__ == "__main__":
    chatbot()