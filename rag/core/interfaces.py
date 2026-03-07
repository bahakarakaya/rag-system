from abc import ABC, abstractmethod
from typing import Optional, Iterator
from .models import *


class DocumentLoader(ABC):
    """Interface for loading raw documents from various sources and converting them into a standardized format for further processing."""
    @abstractmethod
    def load(self, file_path: str) -> Document:
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
    def save(self, embedded_chunks: list[EmbeddedChunk], metadata) -> None:
        ...
    
    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        ...


class Llm(ABC):
    """Interface for interacting with large language models to generate responses based on retrieved chunks."""
    @abstractmethod
    def generate(self, query: str, retrieved_chunks: list[Chunk]) -> str:
        """Generate a response to the query grounded in the provided context chunks."""
        ...