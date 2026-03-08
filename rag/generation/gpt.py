from rag.core.interfaces import Llm
from openai import OpenAI


class GptClient(Llm):
    def __init__(self, model_name: str, api_key: str = None) -> None:
        """Initialize the GPT client with the specified model name, prompt, and API key.  
        The prompt should include placeholders for {context} and {query}."""
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
    
    def generate(self, query: str, retrieved_chunks: list, prompt: str = None) -> str:
        context = "\n\n".join([scored_chunk.chunk.content for scored_chunk in retrieved_chunks])
        prompt = prompt.format(context=context, query=query)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content