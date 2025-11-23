from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from new_llm_model import SimpleLLM
import faiss
import requests


class Rag_Pipeline:
    def __init__(self):
        self.split = RecursiveCharacterTextSplitter(
            chunk_size = 250,
            chunk_overlap = 100
        )
        ##will divide the input text into 250 chunks with 100 overlap so nothing is missed
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

        )
        ##using all-MiniLM-L6-v2 as it is a basic sentence transformer from hugging face embeddings
        ##it is trained to create semantic embeddings
        ##it understands meanings and will convert each chunk into a 384 dimentional vectore
        ##similar meanings similar vectors , model is lightweight for CPU
        
        self.vectorstore = None
        ##FAISS is used for storing vector embeddings and performs similarity search and returns top similar search
        self.llm_url = "http://host.docker.internal:8001/generate"
        ##host.docker.internal refers to laptop , this allows docker to call laptop

    
    def add_documents(self,text):
        chunks = self.split.split_text(text)
        ##this will call the split function on input text which will chunk the input text
        docs = []
        for chunk in chunks:
            docs.append(Document(page_content=chunk))

            # Convert each chunk into a LangChain Document object
            
            ##creating FAISS vectorbase , where vectors will be stored and similarity search is perfomed
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
    
    def retreving(self,query,k=2):
        # Convert query to embedding and search FAISS for similar chunks and returns top k chunks
        if self.vectorstore is None:
            raise ValueError("FAISS index is empty , need to add new docs!!")
        results = self.vectorstore.similarity_search(query,k=k)
        return results
    def generate_ans(self,query):
        docs = self.retreving(query,k=2)
        #this will retrive top 2 chunks

        context = ""
        for doc in docs:
            context += doc.page_content + "\n"

        ##this will add the retrived chunks into one line
        # Step 3: Create the final RAG prompt
        final_prompt = f"""
        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        
        response = requests.post(
        self.llm_url,
        json={"prompt": final_prompt}
    )

        if response.status_code != 200:
            return "LLM server error: " + response.text

        answer = response.json().get("response", "")
        return answer


        



    






    




