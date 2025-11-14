from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

LLM_SERVER_URL = "http://host.docker.internal:8000/ask"  # local endpoint for RAG+LLM

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Forward to your local LLM+RAG service
        response = requests.post(LLM_SERVER_URL, json={"query": query})
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"LLM service error: {response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
