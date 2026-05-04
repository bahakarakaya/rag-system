from rag.core.interfaces import Llm
from openai import OpenAI
from utils.langfuse import get_langfuse_client, start_observation


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

        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="openai.chat.completions",
            as_type="generation",
            model=self.model_name,
            input={"messages": [{"role": "user", "content": prompt}]},
        ) as generation:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
            )
            output = response.choices[0].message.content
            if generation is not None:
                usage = getattr(response, "usage", None)
                usage_details = {}
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)
                    if prompt_tokens is not None:
                        usage_details["input_tokens"] = prompt_tokens
                    if completion_tokens is not None:
                        usage_details["output_tokens"] = completion_tokens
                    if total_tokens is not None:
                        usage_details["total_tokens"] = total_tokens

                update_kwargs = {"output": output}
                if usage_details:
                    update_kwargs["usage_details"] = usage_details
                generation.update(**update_kwargs)

            return output