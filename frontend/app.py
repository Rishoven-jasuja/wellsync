from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.main import chatbot_response

app = Flask(__name__)
CORS(app)  # Allow frontend access

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    
    user_message = data.get("message", "")
    

    # Call your chatbot function from main.py
    bot_response = chatbot_response(user_message)
    

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)