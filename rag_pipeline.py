from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
class RAGPipeline:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        print("Loading embeddings...")
        self.embedder = SentenceTransformer(embedding_model_name)

        self.index = None
        self.documents = []
        self.embeddings = None

    def load_documents(self, folder_path="documents"):
        loader = TextLoader(f"{folder_path}/data.txt")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.documents = splitter.split_documents(docs)


        print(f"Loaded {len(self.documents)} chunks.")
        print("First chunk:", self.documents[0].page_content if self.documents else "NO CHUNKS")


    def build_vectorstore(self):
        print("Building FAISS vector store...")

        texts = [d.page_content for d in self.documents]
        self.embeddings = self.embedder.encode(texts)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query, top_k=3):
        q_emb = self.embedder.encode([query])
        scores, idx = self.index.search(np.array(q_emb), top_k)
        results = [self.documents[i].page_content for i in idx[0]]
        return results


if __name__ == "__main__":
    from llm_model import get_phi_llm

    rag = RAGPipeline()
    rag.load_documents()
    rag.build_vectorstore()

    query = "What does this project talk about?"
    context = rag.retrieve(query)

    llm = get_phi_llm()

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    print(llm.invoke(prompt))
