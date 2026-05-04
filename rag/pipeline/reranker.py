from sentence_transformers import CrossEncoder
import logging


from rag.core.models import Chunk, ScoredChunk
from utils.langfuse import get_langfuse_client, start_observation


logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, cross_encoder: CrossEncoder):
        """
        Initializes the CrossEncoderReranker with a given cross-encoder model.
        Args:
            cross_encoder: An instance of a SentenceTransformer cross-encoder model used for reranking.
        """
        self._cross_encoder = cross_encoder

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            logger.info("No chunks provided for reranking. Returning empty list.")
            return []

        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="cross_encoder.rerank",
            as_type="span",
            input={"query": query, "count": len(chunks), "top_k": top_k},
        ) as rerank_span:
            pairs = [[query, chunk.content] for chunk in chunks]
            scores = self._cross_encoder.predict(pairs) # scores between 0 and 1, higher means more relevant

            reranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            results = [ScoredChunk(chunk=chunk, score=score) for chunk, score in reranked_chunks][:top_k]
            if rerank_span is not None:
                rerank_span.update(output={"result_count": len(results)})
            return results