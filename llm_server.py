##this llm_service creates a light weight microservice using fast api 
##this is needed as the Flask RAG app will run inside docker , while my LLM will run on my laptop
##Docker cannot immport local python files or libraries
##we are exposing LLM as an HHTP api
##What this microservice does:
----------------------------
#1. Loads the LLM model one time at startup using SimpleLLM().
#2. Runs a FastAPI server on port 8001.
#3. Exposes an endpoint: POST /generate
#4. When the Flask app (inside Docker) sends a prompt,
#   this server generates the answer using the local LLM.
#5. Returns the generated text back to Docker as JSON.
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# import your existing LLM wrapper
from new_llm_model import SimpleLLM

# create the FastAPI app
app = FastAPI()

# load your LLM once at startup
llm = SimpleLLM()

# request body schema
class PromptRequest(BaseModel):
    prompt: str

# endpoint for generating text
@app.post("/generate")
def generate_text(request: PromptRequest):
    response = llm.lets_chat(request.prompt)
    return {"response": response}

# start the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
