from rag.core.interfaces import Chunk, Chunker, Document, ChunkMetadata

class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size=500, overlap=75, strip_whitespace: bool = True):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size=chunk_size
        self.overlap=overlap
        self.strip_whitespace=strip_whitespace

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Splits text into chunks of characters based on character count, with optional overlap and whitespace stripping. Simple and fast approach.
        """

        chunks = []
        chunk_index = 0
        step = self.chunk_size - self.overlap

        text = document.content
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            
            if not chunk_text:
                continue

            chunks.append(Chunk(
                content=chunk_text,
                metadata=ChunkMetadata(
                    **document.metadata,
                    chunk_index=chunk_index,
                    start_index=start,
                    end_index=end
                )
            ))
            chunk_index += 1

            if end >= len(text):
                break
        
        return chunks