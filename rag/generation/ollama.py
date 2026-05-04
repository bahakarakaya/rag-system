from rag.core.interfaces import Llm
import requests
import json
from utils.langfuse import get_langfuse_client, start_observation


class OllamaClient(Llm):
    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
    
    def generate(self, query: str, retrieved_chunks: list, prompt: str) -> str:
        context = "\n\n".join([chunk.content for chunk in retrieved_chunks])
        prompt = prompt.format(context=context, query=query)

        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="ollama.generate",
            as_type="generation",
            model=self.model_name,
            input={"prompt": prompt},
        ) as generation:
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
            output = response.json().get("response")
            if generation is not None:
                generation.update(output=output)
            return output