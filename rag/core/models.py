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
class ChunkMetadata(DocumentMetadata):
    chunk_index: int = 0
    start_index: int = 0
    end_index: int = 0

@dataclass
class Document:
    content: str
    metadata: DocumentMetadata

@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]