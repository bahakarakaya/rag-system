from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder
from typing import TYPE_CHECKING

from rag.core.models import Chunk, ScoredChunk
from rag.core.interfaces import VectorStore, Embedder
from rag.stores import BM25Store

if TYPE_CHECKING:
    from rag.pipeline import CrossEncoderReranker


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store, embedder: Embedder, reranker: CrossEncoderReranker):
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        emb_query = self._embedder.embed(query)[0].vector

        vector_hits = self._vector_store.search(emb_query, top_k=top_k*4)
        bm25_hits = self._bm25_store.search(query, top_k=top_k*4)
        sorted_scored_chunks = self._reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
        
        sorted_scored_chunks = self._reranker.rerank(query, [sc.chunk for sc in sorted_scored_chunks], top_k=top_k*2)
        return sorted_scored_chunks[:top_k]
    
    def _reciprocal_rank_fusion(self, vector_hits: list[ScoredChunk], bm25_hits: list[ScoredChunk], k: int = 60) -> list[ScoredChunk]:
        """
        Combines BM25 and vector search results using Reciprocal Rank Fusion (RRF) to produce a final ranked list of chunks.
        
        Args:
            vector_hits: List of ScoredChunk from vector search, sorted by relevance.
            bm25_hits: List of ScoredChunk from BM25 search, sorted by relevance.
            k: The RRF parameter that controls the influence of lower-ranked hits (default=60).
        """
        scores: dict[str, float] = {}
        chunks: dict[tuple, Chunk] = {}

        for rank, scored_chunk in enumerate(vector_hits):
            key = (scored_chunk.chunk.metadata.source, scored_chunk.chunk.metadata.chunk_index)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            chunks[key] = scored_chunk.chunk
        
        for rank, scored_chunk in enumerate(bm25_hits):
            key = (scored_chunk.chunk.metadata.source, scored_chunk.chunk.metadata.chunk_index)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            chunks[key] = scored_chunk.chunk

        sorted_chunks = sorted(chunks.values(), key=lambda c: scores[(c.metadata.source, c.metadata.chunk_index)], reverse=True)
        return [ScoredChunk(chunk=chunk, score=scores[(chunk.metadata.source, chunk.metadata.chunk_index)]) for chunk in sorted_chunks]