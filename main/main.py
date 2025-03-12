import os
import faiss
import json
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to load dataset from multiple JSON files
def load_dataset(data_folder="../data"):
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
                    
                    # Ensure the data is a list of dictionaries
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict) and "question" in entry and "response" in entry:
                                dataset.append(entry)  # Add to dataset
                                intents[entry["question"]] = entry.get("intent", "unknown")  # Store intent
                            else:
                                print(f"Skipping invalid entry in {file_path}: {entry}")  # Debugging info
                    else:
                        print(f"Error: {file_path} does not contain a list. Skipping.")
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

if question_embeddings:
    question_embeddings = np.array(question_embeddings).astype("float32")
    index.add(question_embeddings)  # Add all embeddings to FAISS
else:
    print("Warning: No valid questions found for indexing.")

# Conversation memory (stores last 5 messages)
conversation_memory = []
last_topic = None  # Track the last recognized topic
last_question = None  # Track the last user question

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

def chatbot_response(user_input):
    global last_topic, last_question

    # Preprocess user input
    user_input = preprocess(user_input)

    # Check if input is vague
    if len(user_input.split()) < 4 or any(word in user_input.split() for word in ["its", "their", "them"]):
        user_input = infer_context(user_input)  # Add context

    # Retrieve response
    response, matched_question = get_response(user_input)

    # Store conversation memory (limit to last 5 interactions)
    conversation_memory.append({"user": user_input, "bot": response})
    if len(conversation_memory) > 5:
        conversation_memory.pop(0)

    if matched_question:
        last_topic = matched_question  # Store matched question as last topic
        last_question = user_input  # Store last full question

    return response  # Return response instead of printing


# Retrieve response from FAISS
def get_response(user_input):
    if len(data) == 0:
        return "Sorry, I have no data to answer your query.", None

    query_embedding = model.encode([user_input]).astype("float32")

    _, top_match = index.search(query_embedding, 1)  # Get best match index
    match_index = top_match[0][0]

    if match_index >= 0 and match_index < len(data):
        response_list = data[match_index].get("response", [])
        if response_list:
            return random.choice(response_list), questions[match_index]  # Return response & matched question

    return "I'm not sure. Can you clarify what you're asking about?", None  # No match found

# Infer context from conversation memory
def infer_context(user_input):
    global last_topic, last_question
    vague_keywords = ["its", "their", "them", "that", "this", "these"]

    # If user input contains vague references, append last topic
    if any(word in user_input.split() for word in vague_keywords):
        if last_topic:
            user_input = last_topic + " " + user_input  # Append last topic to new query
        elif last_question:
            user_input = last_question + " " + user_input  # Use last full question as context
    return user_input

# Chatbot function
def chatbot():
    global last_topic, last_question
    print("Chatbot: Hi! How can I help you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Chatbot: Take care! Have a great day.")
            break

        # Preprocess user input
        user_input = preprocess(user_input)

        # Check if input is vague
        if len(user_input.split()) < 4 or any(word in user_input.split() for word in ["its", "their", "them"]):
            user_input = infer_context(user_input)  # Add context

        # Retrieve response
        response, matched_question = get_response(user_input)

        # Store conversation memory (limit to last 5 interactions)
        conversation_memory.append({"user": user_input, "bot": response})
        if len(conversation_memory) > 5:
            conversation_memory.pop(0)

        if matched_question:
            last_topic = matched_question  # Store matched question as last topic
            last_question = user_input  # Store last full question

        print(f"Chatbot: {response}")

# Run chatbot
if __name__ == "__main__":
    chatbot()
