from rank_bm25 import BM25Okapi
import nltk

from rag.core.models import Chunk, ScoredChunk


class BM25Store:
    """Sparse keyword-based retrieval store using the BM25Okapi ranking algorithm."""

    def __init__(self, language: str = 'english'):
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None
        self._language = language
        nltk.download('punkt_tab', quiet=True)

    def save(self, chunks: list[Chunk]) -> None:
        """Replaces the current index with the provided chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        if not chunks:
            raise ValueError("Cannot save an empty list of chunks.")

        self._chunks = chunks
        tokenized_corpus = [nltk.word_tokenize(chunk.content.lower(), language=self._language) for chunk in self._chunks]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Retrieves the top-k chunks most relevant to the query using BM25 scoring.

        Args:
            query: The search query string.
            top_k: Number of top results to return.
        Returns:
            List of ScoredChunk sorted by relevance descending.
        """
        if self._index is None:
            raise RuntimeError("BM25Store is not indexed. Call save() before search().")
        if not query.strip():
            raise ValueError("Query must not be empty.")

        tokenized_query = nltk.word_tokenize(query.lower(), language=self._language)
        scores = self._index.get_scores(tokenized_query)
        scored_chunks = [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in range(len(self._chunks))]
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]