# llm_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading LLM model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # HF pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200
        )

    def generate(self, prompt):
        """Generate a text response."""
        result = self.pipeline(prompt)
        return result[0]["generated_text"]


