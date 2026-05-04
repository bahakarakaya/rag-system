from rag.core.interfaces import Embedder
from rag.core.models import Chunk, EmbeddedChunk
from sentence_transformers import SentenceTransformer
import torch
import logging
from utils.langfuse import get_langfuse_client, start_observation

logger = logging.getLogger(__name__)

class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, chunks: list[Chunk] | str) -> list[EmbeddedChunk]:
        logger.info(f"Embedding {len(chunks)} chunks using model '{self.model_name}' on device '{self.model.device}' with dimension {self.dim}")
        if isinstance(chunks, str):
            chunks = [Chunk(content=chunks, metadata=None)]

        langfuse = get_langfuse_client()
        with start_observation(
            langfuse,
            name="embed_chunks",
            as_type="span",
            input={
                "count": len(chunks),
                "model": self.model_name,
                "device": str(self.device),
            },
        ) as embed_span:
            contents = [chunk.content for chunk in chunks]
            vectors = self.model.encode(contents, precision="float32", convert_to_numpy=True, normalize_embeddings=True)
            embedded_chunks = []
            for chunk, vector in zip(chunks, vectors):
                embedded_chunks.append(EmbeddedChunk(
                    chunk=chunk,
                    vector=vector
                ))

            if embed_span is not None:
                embed_span.update(output={"dimension": self.dim})

            return embedded_chunks