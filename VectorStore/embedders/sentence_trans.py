from rag.core.interfaces import Embedder
from rag.core.models import EmbeddedChunk
from sentence_transformers import SentenceTransformer

class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, chunks):
        print(f"Embedding {len(chunks)} chunks using model '{self.model_name}' on device '{self.model.device}' with dimension {self.dim}")
        contents = [chunk.content for chunk in chunks]
        vectors = self.model.encode(contents)
        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(EmbeddedChunk(
                chunk=chunk,
                vector=vector.tolist()
            ))

        

        return embedded_chunks