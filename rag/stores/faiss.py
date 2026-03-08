from rag.core.interfaces import VectorStore
from rag.core.models import Chunk, ChunkMetadata, EmbeddedChunk, ScoredChunk
from datetime import datetime
from utils.db import DatabaseManager
import numpy as np
from pathlib import Path
import faiss
import logging

logger = logging.getLogger(__name__)

class FaissVectorStore(VectorStore):
    def __init__(self, index_type: str = "IndexFlatIP", index_path: str = None, dimension: int = 384, gpu: bool = False, db_path: str = None, index = None):
        self.index_type = index_type
        self.index_path = index_path
        self.dimension = dimension
        self.gpu = gpu
        self.next_id = 0
        self.db_manager = DatabaseManager(db_path)
        self.db_manager.create_metadata_table()

        if db_path is None:
            db_path = Path.cwd() / "data" / "index_dir" / "vector_store_metadata.db"
        else:
            db_path = Path(db_path)

        if index is not None:
            # Use pre-loaded index object
            self.index = index
            self.dimension = index.d
            self.index_type = index.__class__.__name__
            self.next_id = self.db_manager.get_max_id() + 1
        elif index_path is not None and Path(index_path).exists():
            # Load index from disk
            loaded = faiss.read_index(index_path)
            self.index = loaded
            self.dimension = loaded.d
            self.index_type = loaded.__class__.__name__
            self.next_id = self.db_manager.get_max_id() + 1
        else:
            # Create new index
            _index_class_map = {
                "IndexFlatIP": faiss.IndexFlatIP,
                "IndexFlatL2": faiss.IndexFlatL2,
            }
            if index_type in _index_class_map:
                faiss_index = faiss.IndexIDMap(_index_class_map[index_type](dimension))
            else:
                faiss_index = faiss.index_factory(dimension, index_type)
            
            if gpu: # Not compatible with python 3.12 yet, I will build it from source later or try with 3.10
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
                self.index = gpu_index
            else:
                self.index = faiss_index

    @classmethod
    def load(cls, index_path: str, db_path: str = None) -> 'FaissVectorStore':
        """Loads a FaissVectorStore from the specified index path and database path. If the index file does not exist, a new FaissVectorStore will be created with the provided database path.
        Args:
            index_path (str): The file path to load the Faiss index from.
            db_path (str, optional): The file path to the SQLite database for metadata.
        Returns:
            FaissVectorStore: An instance of FaissVectorStore with the loaded index and database connection.
        """
        index = faiss.read_index(index_path)
        return cls(index=index, index_path=index_path, db_path=db_path)
    
    def _validate_query_vector(self, query_vector: np.ndarray) -> np.ndarray:
        if not isinstance(query_vector, np.ndarray):
            raise TypeError(f"query_vector must be a np.ndarray, got {type(query_vector)}")
        if query_vector.dtype != np.float32:
            raise ValueError(f"query_vector must be float32, got {query_vector.dtype}")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"query_vector dimension {query_vector.shape[1]} does not match index dimension {self.dimension}")
        return query_vector

    def save(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Saves the embedded chunks to the Faiss index and metadata to the database. If any error occurs during the process, the database transaction is rolled back to maintain consistency.
        Args:
            embedded_chunks (list[EmbeddedChunk]): A list of EmbeddedChunk objects to be saved.
                Each chunk's metadata may include a content_hash from the source document.
        Raises:
            Exception: Any exception that occurs during the database insertion or Faiss index update will be raised.
        """
        if not embedded_chunks:
            return

        try:
            ids = np.arange(self.next_id, self.next_id + len(embedded_chunks), dtype=np.int64)

            vectors = np.empty((len(embedded_chunks), self.dimension), dtype=np.float32)
            for i, embedded_chunk in enumerate(embedded_chunks):
                vectors[i] = embedded_chunk.vector

            metadatas = [{
                    "id": int(ids[i]),
                    "content": embedded_chunk.chunk.content,
                    "source": embedded_chunk.chunk.metadata.source,
                    "doc_id": embedded_chunk.chunk.metadata.doc_id,
                    "format": embedded_chunk.chunk.metadata.format,
                    "chunk_index": embedded_chunk.chunk.metadata.chunk_index,
                    "start_index": embedded_chunk.chunk.metadata.start_index,
                    "end_index": embedded_chunk.chunk.metadata.end_index,
                    "section": embedded_chunk.chunk.metadata.section,
                    "page": embedded_chunk.chunk.metadata.page,
                    "created_at": embedded_chunk.chunk.metadata.created_at.isoformat() if embedded_chunk.chunk.metadata.created_at else None,
                    "content_hash": embedded_chunk.chunk.metadata.content_hash
                }
                for i, embedded_chunk in enumerate(embedded_chunks)
            ]

            self.db_manager.insert_metadata(metadatas)
            self.index.add_with_ids(vectors, ids)

            self.db_manager.conn.commit()
            self.next_id = ids[-1] + 1
            self.persist()

        except Exception as e:
            self.db_manager.conn.rollback()
            raise

    def remove_vectors(self, ids: list[int]) -> None:
        """Remove vectors from the FAISS index by their IDs and persist the updated index.

        Args:
            ids: List of vector IDs to remove.
        """
        if not ids:
            return
        id_array = np.array(ids, dtype=np.int64)
        self.index.remove_ids(id_array)
        self.persist()

    def persist(self, index_path: str = None) -> None:
        path = index_path or self.index_path
        if path is None:
            raise ValueError("index_path must be provided either at construction or at persist() call")
        faiss.write_index(self.index, path)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        query_vector = self._validate_query_vector(query_vector)
        distances, ids = self.index.search(query_vector, top_k)
        results = []
        for dist, chunk_id in zip(distances[0], ids[0]):
            if chunk_id == -1:
                continue
            metadata = self.db_manager.get_metadata_by_id(chunk_id)
            if metadata is not None:
                raw_dt = metadata["created_at"]
                created_at = datetime.fromisoformat(raw_dt) if isinstance(raw_dt, str) else raw_dt
                chunk = Chunk(
                    content=metadata["content"],
                    metadata=ChunkMetadata(
                        source=metadata["source"],
                        doc_id=metadata["doc_id"],
                        format=metadata["format"],
                        chunk_index=metadata["chunk_index"],
                        start_index=metadata["start_index"],
                        end_index=metadata["end_index"],
                        section=metadata["section"],
                        page=metadata["page"],
                        created_at=created_at,
                        content_hash=metadata.get("content_hash")
                    )
                )
                results.append(ScoredChunk(chunk=chunk, score=float(dist)))
            else:
                logger.warning(f"No metadata found for id {chunk_id}")
        return results