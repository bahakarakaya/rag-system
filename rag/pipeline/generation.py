from rag.core.interfaces import Llm
from rag.pipeline.query import QueryPipeline
from utils.langfuse import get_langfuse_client, start_observation


class GenerationPipeline:
    def __init__(self, llm: Llm, query_pipeline: QueryPipeline, prompt: str) -> None:
        self.llm = llm
        self.query_pipeline = query_pipeline
        self.prompt = prompt

    def run(self, query: str, top_k: int = 5) -> dict:
        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="generation_pipeline.run",
            as_type="span",
            input={"query": query, "top_k": top_k},
        ) as generation_span:
            retrieved_scored_chunks = self.query_pipeline.run(query_text=query, top_k=top_k)
            chunk_sources = [scored_chunk.chunk.metadata.source for scored_chunk in retrieved_scored_chunks] if retrieved_scored_chunks else ["unknown"]
            answer = self.llm.generate(
                query=query,
                retrieved_chunks=retrieved_scored_chunks,
                prompt=self.prompt
            )
            if generation_span is not None:
                generation_span.update(output={"answer": answer, "sources": chunk_sources})
            return {"answer": answer, "sources": chunk_sources, "chunks": retrieved_scored_chunks}