import json
import os
from sentence_transformers import SentenceTransformer, util

class Chatbot:
    def __init__(self, memory_file="chat_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()  # Load past conversation history
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def load_memory(self):
        """Load conversation history from a JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as file:
                return json.load(file)
        return {}

    def save_memory(self):
        """Save conversation history to a JSON file."""
        with open(self.memory_file, "w") as file:
            json.dump(self.memory, file, indent=4)

    def store_message(self, user, message, response):
        """Store user messages and chatbot responses in memory."""
        if user not in self.memory:
            self.memory[user] = []
        
        # Store message as a dictionary (text, response)
        self.memory[user].append({"message": message, "response": response})
        self.save_memory()  # Save to JSON file

    def get_contextual_response(self, user, new_message):
        """Retrieve the most relevant past message for context-based response."""
        if user not in self.memory or not self.memory[user]:
            return None

        new_embedding = self.model.encode(new_message, convert_to_tensor=True)

        best_match = None
        highest_score = -1

        for entry in self.memory[user]:
            past_message = entry["message"]
            past_response = entry["response"]
            past_embedding = self.model.encode(past_message, convert_to_tensor=True)

            similarity_score = util.pytorch_cos_sim(new_embedding, past_embedding).item()
            if similarity_score > highest_score:
                highest_score = similarity_score
                best_match = past_response

        return best_match if highest_score > 0.5 else None  # Threshold for relevance

    def generate_response(self, user, message):
        """Generate a response based on conversation history and similarity matching."""
        contextual_response = self.get_contextual_response(user, message)

        if contextual_response:
            response = contextual_response
        else:
            # Default responses if no relevant past query is found
            if "stress" in message.lower():
                response = "I'm sorry you're feeling stressed. Would you like some tips to manage it?"
            elif "handle it" in message.lower():
                response = "Here are some ways to handle stress: exercise, meditation, and deep breathing."
            else:
                response = "I'm here to help. Can you provide more details?"

        self.store_message(user, message, response)  # Store conversation history
        return response


# Example Usage
bot = Chatbot()
user_id = "user1"

print(bot.generate_response(user_id, "I am dealing with stress"))  # Stores and responds
print(bot.generate_response(user_id, "Tell me ways to handle it"))  # Uses context for response
print(bot.generate_response(user_id, "Give me more tips"))  # Retrieves relevant past response
