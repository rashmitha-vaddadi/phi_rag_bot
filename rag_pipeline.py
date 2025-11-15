# rag_pipeline.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class RAGPipeline:

    def __init__(self, docs_path="documents/data.txt"):
        print("Initializing RAG pipeline...")

        self.docs_path = docs_path

        # Load embedding model
        print("Loading embeddings model: all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load documents
        self.load_documents()

        # Build vector index
        self.build_vectorstore()

    def load_documents(self):
        print(f"Loading documents from: {self.docs_path}")
        loader = TextLoader(self.docs_path)
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.chunks = splitter.split_documents(docs)

    def build_vectorstore(self):
        print("Building FAISS vector store...")

        texts = [c.page_content for c in self.chunks]
        embeddings = [self.embedder.encode(t).astype(np.float32) for t in texts]

        self.vectorstore = FAISS.from_embeddings(texts, embeddings)
        print("FAISS vectorstore created.")

    def retrieve(self, query, top_k=3):
        print("Retrieving context...")
        q_emb = self.embedder.encode(query).astype(np.float32)

        docs = self.vectorstore.similarity_search_by_vector(q_emb, k=top_k)
        return [d.page_content for d in docs]




