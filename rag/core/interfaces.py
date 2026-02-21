from abc import ABC, abstractmethod, field
from typing import Optional, Iterator
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentMetadata:
    source: str                      # e.g., "file path", "url", "database" where the document was obtained from
    doc_id: str
    format: str
    section: Optional[str] = None    # e.g., "introduction", "conclusion", "chapter 1" to provide more context about the document's content
    page: Optional[int] = None
    created_at: Optional[datetime] = None

@dataclass
class Document:
    content: str
    metadata: DocumentMetadata

@dataclass
class Chunk:
    content: str
    metadata: DocumentMetadata

@dataclass
class EmbeddedChunk(Chunk):
    chunk: Chunk
    vector: list[float]


class DocumentLoader(ABC):
    """Interface for loading raw documents from various sources and converting them into a standardized format for further processing."""
    @abstractmethod
    def load(self) -> Iterator[Document]:
        ...


class Chunker(ABC):
    """Interface for splitting documents into smaller, manageable pieces (chunks) based on specified criteria."""
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...


class Embedder(ABC):
    """Interface for converting chunks into vector representations that can be used for similarity search and retrieval."""
    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        ...


class VectorStore(ABC):
    """Interface for storing and retrieving embedded chunks based on their vector representations."""
    @abstractmethod
    def save(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        ...
    
    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        ...