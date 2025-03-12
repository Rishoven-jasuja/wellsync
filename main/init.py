import os  #working
import faiss
import json
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to load dataset from multiple JSON files without debug prints
def load_dataset(data_folder="data"):
    dataset = []
    intents = {}  # Dictionary to store intents

    if not os.path.exists(data_folder):
        # Folder not found, return empty results silently
        return dataset, intents

    for file in os.listdir(data_folder):
        if file.endswith(".json"):
            file_path = os.path.join(data_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    entries = json.load(f)
                    if isinstance(entries, list):
                        for entry in entries:
                            # Only add valid entries; silently skip invalid ones
                            if isinstance(entry, dict) and "question" in entry and "response" in entry:
                                dataset.append(entry)
                                intents[entry["question"]] = entry.get("intent", "unknown")
                    # If entries is not a list, skip silently
                except json.JSONDecodeError:
                    # Skip files that cannot be parsed
                    pass
    return dataset, intents

# Load dataset and intents
data, intents = load_dataset()

# Create FAISS index using cosine similarity
dimension = 384  # Embedding size for MiniLM
index = faiss.IndexFlatIP(dimension)

# Encode questions & add to FAISS with normalized embeddings
question_embeddings = []
questions = []  # Store questions separately for indexing

for entry in data:
    embedding = model.encode([entry["question"]])[0]
    question_embeddings.append(embedding)
    questions.append(entry["question"])

if question_embeddings:
    question_embeddings = np.array(question_embeddings).astype("float32")
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(question_embeddings)
    index.add(question_embeddings)
else:
    # Optionally, handle case where no embeddings were added
    pass

# Conversation memory (stores last 5 messages)
conversation_memory = []
last_topic = None  # Track the last recognized topic
last_question = None  # Track the last user question

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def chatbot_response(user_input):
    global last_topic, last_question
    user_input = preprocess(user_input)

    # If the query is vague, infer context
    if len(user_input.split()) < 4 or any(word in user_input.split() for word in ["its", "their", "them"]):
        user_input = infer_context(user_input)

    response, matched_question = get_response(user_input)

    # Store conversation memory (limit to last 5 interactions)
    conversation_memory.append({"user": user_input, "bot": response})
    if len(conversation_memory) > 5:
        conversation_memory.pop(0)

    if matched_question:
        last_topic = matched_question
        last_question = user_input

    return response

# Retrieve response using cosine similarity
def get_response(user_input):
    if len(data) == 0:
        return "Sorry, I have no data to answer your query.", None

    query_embedding = model.encode([user_input]).astype("float32")
    faiss.normalize_L2(query_embedding)

    _, top_match = index.search(query_embedding, 1)
    match_index = top_match[0][0]

    if 0 <= match_index < len(data):
        response_list = data[match_index].get("response", [])
        if response_list:
            return random.choice(response_list), questions[match_index]

    return "I'm not sure. Can you clarify what you're asking about?", None

# Infer context from conversation memory
def infer_context(user_input):
    global last_topic, last_question
    vague_keywords = ["its", "their", "them", "that", "this", "these"]

    if any(word in user_input.split() for word in vague_keywords):
        if last_topic:
            user_input = last_topic + " " + user_input
        elif last_question:
            user_input = last_question + " " + user_input
    return user_input

# Chatbot function to interact in the console
def chatbot():
    global last_topic, last_question
    print("Chatbot: Hi! How can I help you today? (Type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Take care! Have a great day.")
            break

        user_input = preprocess(user_input)
        if len(user_input.split()) < 4 or any(word in user_input.split() for word in ["its", "their", "them"]):
            user_input = infer_context(user_input)

        response, matched_question = get_response(user_input)
        conversation_memory.append({"user": user_input, "bot": response})
        if len(conversation_memory) > 5:
            conversation_memory.pop(0)

        if matched_question:
            last_topic = matched_question
            last_question = user_input

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
