# app.py
from flask import Flask, request, jsonify
from rag_pipeline import RAGPipeline
from llm_model import LocalLLM

app = Flask(__name__)

print("Loading RAG pipeline...")
rag = RAGPipeline()

print("Loading Local LLM...")
llm = LocalLLM()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query missing"}), 400

    # 1 — Retrieve relevant chunks
    contexts = rag.retrieve(query)
    context_text = "\n".join(contexts)

    # 2 — Build final prompt
    prompt = f"""
Use the context to answer the question.

Context:
{context_text}

Question: {query}
Answer:
"""

    # 3 — Generate answer
    answer = llm.generate(prompt)

    return jsonify({
        "query": query,
        "answer": answer,
        "context_used": contexts
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
