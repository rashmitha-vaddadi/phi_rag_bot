from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

def get_phi_llm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print("Loading Phi model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    generate_text = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm


if __name__ == "__main__":
    llm = get_phi_llm()
    print(llm.invoke("Explain transformers in simple words."))
