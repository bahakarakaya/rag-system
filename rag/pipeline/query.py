from rag.core.interfaces import Embedder, VectorStore
from rag.core.models import Chunk
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores.faiss import FaissVectorStore
import json
import logging

logger = logging.getLogger(__name__)

class QueryPipeline:
    # query embedder ve ingestion embedder modelleri aynı olmalı. Çünkü farklı modeller farklı embedding tekniklerini kullanır, farklı vektör uzayında yaşarlar.
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store
    
    def run(self, query_text: str, top_k: int = 5):
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
        
        query = Chunk(content=query_text, metadata=None)
        
        try:
            query_vector = self.embedder.embed([query])[0].vector
            results = self.store.search(query_vector, top_k=top_k)
            
            return results
        except Exception as e:
            raise RuntimeError(f"An error occurred during query execution: {str(e)}")