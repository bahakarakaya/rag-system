from rag.core.interfaces import Llm
from rag.pipeline.query import QueryPipeline


class GenerationPipeline:
    def __init__(self, llm: Llm, query_pipeline: QueryPipeline, prompt: str) -> None:
        self.llm = llm
        self.query_pipeline = query_pipeline
        self.prompt = prompt

    def run(self, question: str, top_k: int = 5) -> str:
        retrieved_chunks = self.query_pipeline.run(query_text=question, top_k=top_k)
        chunk_sources = [chunk.metadata.source for chunk in retrieved_chunks] if retrieved_chunks else ["unknown"]
        answer = self.llm.generate(
            query=question,
            retrieved_chunks=retrieved_chunks,
            prompt=self.prompt
        )
        return {"answer": answer, "sources": chunk_sources}