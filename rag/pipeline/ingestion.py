from rag.core.interfaces import DocumentLoader, Chunker, Embedder, VectorStore
from rag.core.models import Document, Chunk, ChunkMetadata
from rag.ingestion.loaders import TextFileLoader
from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores.faiss import FaissVectorStore
from rag.stores.bm25 import BM25Store
from utils.hashing import compute_content_hash
import logging

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the full ingestion process: loading, chunking, embedding, and storing documents into the vector store.

    Args:
        chunker: Chunker object - The chunker instance to use (e.g., "FixedSizeChunker").  
        embedder: Embedder object - The embedder instance to use (e.g., "sentence-transformers").  
        vector_store: VectorStore object - The vector store instance to use (e.g., "FAISS").
        bm25_store: BM25Store object (optional) - When provided, all ingested documents are also
            indexed into the BM25 store to enable hybrid search.
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore, bm25_store: BM25Store | None = None) -> None:
        self._validate_components(chunker, embedder, vector_store, bm25_store)
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._bm25_corpus: dict[str, list] = {}  # source -> chunks; accumulates across run() calls
        if self._bm25_store is not None:
            self._warm_bm25_from_db()
    
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
    
            new_chunks: dict[str, list] = {}
            for doc in all_documents:
                chunks = self._process_document(doc)
                if chunks:
                    new_chunks[doc.metadata.source] = chunks

            if self._bm25_store is not None and new_chunks:
                self._index_bm25(new_chunks)
        except Exception as e:
            raise RuntimeError(f"Ingestion pipeline failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_document(self, doc: Document) -> list:
        """Processes a single document: checks for changes, handles re-indexing if necessary.
        
        Returns the produced chunks so they can be reused for BM25 indexing.
        Returns an empty list if the document is unchanged and was skipped.
        """
        source = doc.metadata.source
        current_hash = compute_content_hash(doc.content)
        is_indexed = self._vector_store.db_manager.is_document_indexed(source)
        stored_hash = self._vector_store.db_manager.get_stored_hash(source)

        if is_indexed and (stored_hash == current_hash):
            logger.info(f"Document '{source}' is already indexed and unchanged, skipping.")
            return []

        if is_indexed:
            logger.info(f"Document '{source}' has changed, re-indexing...")
            stale_ids = self._vector_store.db_manager.delete_by_source(source)
            self._vector_store.remove_vectors(stale_ids)
        else:
            logger.info(f"New document '{source}', indexing...")

        doc.metadata.content_hash = current_hash
        return self._ingest_document(doc)

    def _ingest_document(self, doc: Document) -> list:
        """Internal helper to chunk, embed, and save a single document. Returns the produced chunks."""
        chunks = self._chunker.chunk(doc)
        logger.info(f"Document: {doc.metadata.doc_id} - {len(chunks)} chunks created.")

        for chunk in chunks:
            logger.debug(f"-------------------- CHUNK --------------------\n{chunk.metadata}: {chunk.content}")

        logger.info(f"Embedding chunks for document '{doc.metadata.doc_id}'...")
        embedded_chunks = self._embedder.embed(chunks)
        for embedded_chunk in embedded_chunks:
            logger.debug(f"Embedded Chunk Metadata: {embedded_chunk.chunk.metadata}")
            logger.debug(f"Embedded Chunk Vector (first 3 dimensions): {embedded_chunk.vector[:3]}")

        self._vector_store.save(embedded_chunks)
        return chunks

    def _resolve_loader(self, source) -> DocumentLoader:
        """Determines the appropriate loader for a given source based on its file extension."""
        if source.endswith(".txt"):
            return TextFileLoader()
        else:
            raise ValueError(f"No loader available for source: {source}. Available loaders: .txt")
        

    def _warm_bm25_from_db(self) -> None:
        """Populates the BM25 corpus and index from chunks already persisted in the vector store DB.

        Called once on init so that queries work even when all documents are unchanged
        and _index_bm25 is never triggered during the current process.
        """
        rows = self._vector_store.db_manager.get_all_chunks()
        if not rows:
            return

        for row in rows:
            chunk = Chunk(
                content=row["content"],
                metadata=ChunkMetadata(
                    source=row["source"],
                    doc_id=row["doc_id"],
                    format=row["format"],
                    chunk_index=row["chunk_index"],
                    start_index=row["start_index"],
                    end_index=row["end_index"],
                    section=row["section"],
                    page=row["page"],
                    created_at=row["created_at"],
                )
            )
            self._bm25_corpus.setdefault(row["source"], []).append(chunk)

        all_chunks = [c for chunks in self._bm25_corpus.values() for c in chunks]
        self._bm25_store.save(all_chunks)
        logger.info(f"BM25 warmed from DB: {len(all_chunks)} chunks from {len(self._bm25_corpus)} document(s).")

    def _index_bm25(self, new_chunks: dict[str, list]) -> None:
        """Updates the BM25 corpus with pre-computed chunks and rebuilds the full index.

        Accepts chunks already produced during FAISS ingestion to avoid re-chunking.
        The corpus dict maps source -> chunks and persists across run() calls so that
        incremental ingestion does not evict previously indexed documents from the
        in-memory BM25 index.
        """
        for source, chunks in new_chunks.items():
            self._bm25_corpus[source] = chunks
            logger.info(f"BM25: updated {len(chunks)} chunks for '{source}'.")

        all_chunks = [chunk for chunks in self._bm25_corpus.values() for chunk in chunks]
        self._bm25_store.save(all_chunks)
        logger.info(f"BM25 index rebuilt with {len(all_chunks)} total chunks from {len(self._bm25_corpus)} document(s).")

    def _validate_components(self, chunker, embedder, store, bm25_store):
        """Validates that the provided components implement the required interfaces."""
        if not isinstance(chunker, Chunker):
            raise TypeError(f"chunker must implement Chunker, got {type(chunker).__name__}")
        if not isinstance(embedder, Embedder):
            raise TypeError(f"embedder must implement Embedder, got {type(embedder).__name__}")
        if not isinstance(store, VectorStore):
            raise TypeError(f"store must implement VectorStore, got {type(store).__name__}")
        if bm25_store is not None and not isinstance(bm25_store, BM25Store):
            raise TypeError(f"bm25_store must be a BM25Store instance, got {type(bm25_store).__name__}")