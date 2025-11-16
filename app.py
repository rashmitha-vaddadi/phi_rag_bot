##flask api will act as an interface between user and rag+llm

from flask import Flask, request, jsonify
from new_rag_pipeline import Rag_Pipeline
import os

app = Flask(__name__)
rag = Rag_Pipeline()

## ----------------------
## AUTO-LOAD DOCUMENTS ON STARTUP
## ----------------------

DOCUMENTS_FOLDER = "documents"

for filename in os.listdir(DOCUMENTS_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            rag.add_documents(text)
            print(f"Loaded: {filename}")


## ----------------------
## INGEST ENDPOINT
## ----------------------

@app.route("/ingest", methods=["POST"])
def ingest():
    text1 = request.get_json()
    input_text = text1.get("text")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    rag.add_documents(input_text)
    return jsonify({"message": "Documents added successfully"})


## ----------------------
## QUERY ENDPOINT
## ----------------------

@app.route("/query", methods=["POST"])
def query():
    text2 = request.get_json()
    question = text2.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = rag.generate_ans(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


