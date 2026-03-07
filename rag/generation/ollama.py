from rag.core.interfaces import Llm
import requests
import json


class OllamaClient(Llm):
    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
    
    def generate(self, query: str, retrieved_chunks: list, prompt: str) -> str:
        context = "\n\n".join([chunk.content for chunk in retrieved_chunks])
        prompt = prompt.format(context=context, query=query)
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 512,
                    "temperature": 0
                }
            }
        )
        response.raise_for_status()
        return response.json().get("response")