from rag.core.interfaces import DocumentLoader, Chunker, Embedder, VectorStore
from rag.core.models import Document
from rag.ingestion.loaders import TextFileLoader
from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores.faiss import FaissVectorStore
from utils.hashing import compute_content_hash
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the full ingestion process: loading, chunking, embedding, and storing documents into the vector store.

    Args:
        chunker: Chunker object - The chunker instance to use (e.g., "FixedSizeChunker").  
        embedder: Embedder object - The embedder instance to use (e.g., "sentence-transformers").  
        store: VectorStore object - The vector store instance to use (e.g., "FAISS").
        device: str - The device to use for embedding (e.g., "cuda" or "cpu").
        config_path: str - The path to save the pipeline configuration for use by the QueryPipeline
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, store: VectorStore) -> None:
        CONFIG_PATH = Path("/home/bhkrky/ws/rag-system/data/config.json")
        self._validate_components(chunker, embedder, store)
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._config_path = CONFIG_PATH
        self._save_config()
    
    def run(self, source_paths: list[str]):
        """
        Adds documents to the vector store.

        Args:
            source_paths: List of document sources (e.g., file paths).
        
        Raises:
            ValueError: If no documents are provided or if an unsupported document type is encountered.
            RuntimeError: If any step in the ingestion process fails.

        Example:
            Direct list:  
                pipeline.run(source_paths=[
                "/docs/report.pdf",
                "/docs/notes.txt"
                ])
            
            Unpacked list:
                pipeline.run(source_paths=[*doc_list])
        """
        if not source_paths:
            raise ValueError("No document path provided for ingestion.")

        try:
            all_documents = []
            for source in source_paths:
                loader = self._resolve_loader(source)
                doc = loader.load(source)
                all_documents.append(doc)
    
            for doc in all_documents:
                self._process_document(doc)
        except Exception as e:
            raise RuntimeError(f"Ingestion pipeline failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_document(self, doc: Document):
        """Processes a single document: checks for changes, handles re-indexing if necessary, and returns the document ready for chunking."""
        source = doc.metadata.source
        current_hash = compute_content_hash(doc.content)
        is_indexed = self._store.db_manager.is_document_indexed(source)
        stored_hash = self._store.db_manager.get_stored_hash(source)

        if is_indexed and (stored_hash == current_hash):
            logger.info(f"Document '{source}' is already indexed and unchanged, skipping.")
            return

        if is_indexed:
            logger.info(f"Document '{source}' has changed, re-indexing...")
            stale_ids = self._store.db_manager.delete_by_source(source)
            self._store.remove_vectors(stale_ids)
        else:
            logger.info(f"New document '{source}', indexing...")
        
        doc.metadata.content_hash = current_hash

        self._ingest_document(doc)

    def _ingest_document(self, doc: Document):
        """Internal helper to chunk, embed, and save a single document."""
        chunks = self._chunker.chunk(doc)
        logger.info(f"Document: {doc.metadata.doc_id} - {len(chunks)} chunks created.")

        for chunk in chunks:
            logger.debug(f"-------------------- CHUNK --------------------\n{chunk.metadata}: {chunk.content}")

        logger.info(f"Embedding chunks for document '{doc.metadata['doc_id']}'...")
        embedded_chunks = self._embedder.embed(chunks)
        for embedded_chunk in embedded_chunks:
            logger.debug(f"Embedded Chunk Metadata: {embedded_chunk.chunk.metadata}")
            logger.debug(f"Embedded Chunk Vector (first 3 dimensions): {embedded_chunk.vector[:3]}")

        self._store.save(embedded_chunks)

    def _save_config(self):
        """Saves the pipeline configuration to disk to ensure QueryPipeline can use the same components for querying."""
        config = {
            "embedder_model": self._embedder.model_name,
            "chunker_type": self._chunker.__class__.__name__,
            "store_type": self._store.__class__.__name__,
            "device": getattr(self._embedder, "device")
        }

        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        if self._config_path.exists():
            with self._config_path.open("r") as f:
                existing_config = json.load(f)
            if existing_config.get("embedder_model") != self._embedder.model_name:
                logger.warning(f"Overwriting existing config. Previous model: {existing_config.get('embedder_model')}, New model: {self._embedder.model_name}")
            if existing_config.get("chunker_type") != self._chunker.__class__.__name__:
                logger.warning(f"Overwriting existing config. Previous chunker: {existing_config.get('chunker_type')}, New chunker: {self._chunker.__class__.__name__}")
            if existing_config.get("store_type") != self._store.__class__.__name__:
                logger.warning(f"Overwriting existing config. Previous store: {existing_config.get('store_type')}, New store: {self._store.__class__.__name__}")

        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=4)

    def _resolve_loader(self, source) -> DocumentLoader:
        """Determines the appropriate loader for a given source based on its file extension."""
        if source.endswith(".txt"):
            return TextFileLoader()
        else:
            raise ValueError(f"No loader available for source: {source}. Available loaders: .txt")
        

    def _validate_components(self, chunker, embedder, store):
        """Validates that the provided components implement the required interfaces."""
        if not isinstance(chunker, Chunker):
            raise TypeError(f"chunker must implement Chunker, got {type(chunker).__name__}")
        if not isinstance(embedder, Embedder):
            raise TypeError(f"embedder must implement Embedder, got {type(embedder).__name__}")
        if not isinstance(store, VectorStore):
            raise TypeError(f"store must implement VectorStore, got {type(store).__name__}")