from flask import Flask, request, jsonify
from flask_cors import CORS
from main/main.py import chatbot_response  # Correct import from main/main.py

app = Flask(__name__)
CORS(app)  # Allow frontend access

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_message = data.get("message", "")

    # Call your chatbot function from main.py
    bot_response = chatbot_response(user_message)

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
