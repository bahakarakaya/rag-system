from __future__ import annotations

from typing import TYPE_CHECKING
from rag.core.models import ScoredChunk
import logging

if TYPE_CHECKING:
    from rag.stores import HybridRetriever

logger = logging.getLogger(__name__)

class QueryPipeline:
    # query embedder ve ingestion embedder modelleri aynı olmalı. Çünkü farklı modeller farklı embedding tekniklerini kullanır, farklı vektör uzayında yaşarlar.
    def __init__(self, retriever: HybridRetriever) -> None:
        self.hybrid_retriever = retriever

    def run(self, query_text: str, top_k: int = 5) -> list[ScoredChunk]:
        """
        Executes the query pipeline: embeds the query, retrieves relevant documents from the vector store, and returns them.

        Args:
            query_text: The input query string to search for.
            top_k: The number of top relevant documents to retrieve from the vector store.

        Returns:
            List of retrieved documents relevant to the query.

        Raises:
            ValueError: If the query is empty or if any component is not properly initialized.
            RuntimeError: If any step in the query process fails.
        """
        if not query_text.strip():
            raise ValueError("Query cannot be empty.")
        
        try:
            result_chunks = self.hybrid_retriever.retrieve(query_text, top_k=top_k)
            return result_chunks
        except Exception as e:
            raise RuntimeError(f"An error occurred during query execution: {str(e)}")