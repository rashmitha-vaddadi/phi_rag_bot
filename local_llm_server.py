from flask import Flask, request, jsonify
from rag_pipeline import RAGPipeline
from llm_model import get_phi_llm

# Initialize Flask app
app = Flask(__name__)

print("🚀 Loading RAG pipeline and LLM model...")
rag = RAGPipeline()
llm = get_phi_llm()

@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint to handle questions and return RAG + LLM answers."""
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Step 1: Retrieve context using RAG
    context_chunks = rag.retrieve(query)
    context = "\n".join(context_chunks)

    # Step 2: Create prompt for LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Step 3: Generate response
    try:
        answer = llm.invoke(prompt)
        return jsonify({
            "query": query,
            "answer": answer,
            "context_used": context_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

