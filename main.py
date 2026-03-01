from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores.faiss import FaissVectorStore
from rag.pipeline import IngestionPipeline, QueryPipeline
from rag.core.models import Chunk
from utils.hashing import compute_content_hash
from time import time
from pathlib import Path
import logging
import sys
import os

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=logging.WARNING,  # suppress third-party library logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Apply the desired level only to our application namespaces
for _ns in ("__main__", "rag", "VectorStore", "utils"):
    logging.getLogger(_ns).setLevel(_log_level)

logger = logging.getLogger(__name__)
config_path = Path("data/config.json")


def main():

    query_text = "What are the main components of a RAG system and how do they interact with each other?"
    query_text_irrelevant = "How is the weather like in Izmir today?"

    logger.info("RAG system starting...")

    embedder = SentenceTransformersEmbedder()
    chunker = FixedSizeChunker()
    if not Path("data/index_dir/faiss_index.bin").exists():
        store = FaissVectorStore(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")
    else:
        store = FaissVectorStore.load(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")

    ingestion_pipe = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        store=store
    )
    query_pipe = QueryPipeline(
        embedder=embedder,
        store=store
    )

    ingestion_pipe.run(source_paths=["data/ai_eng_prj.txt"])

    results = query_pipe.run(query_text=query_text, top_k=3)
    
    if not results:
        logger.info("No results found for the relevant query. Returning empty list.")
    else:
        logger.debug(f"Search results for query: '{query_text}'")
        for i, result in enumerate(results):
            logger.debug(f"RESULT {i+1}:")
            logger.debug(f"CONTENT: {result.content.strip()}")
            logger.debug(f"METADATA: {result.metadata}\n")

    #---------------------------------------------------------------------------------------------------------------------

    
    """ print("\nPerforming search with relevant query...")
    query_text = Chunk(content=query_text, metadata=None)
    query_vector = embedder.embed([query_text])[0].vector
    print(f"Query Vector (first 3 dimensions): {query_vector[:3]}")
    results = store.search(query_vector, top_k=2)
    print(f"Search results for query: '{query_text.content}'")
    if not results:
        print("No results found for the relevant query.")
    else:
        for i, result in enumerate(results):
            print(f"RESULT {i+1}:")
            print(f"CONTENT: {result.content}")
            print(f"METADATA: {result.metadata}\n")
    
    print("\nPerforming search with irrelevant query...")
    query_text_irrelevant = Chunk(content=query_text_irrelevant, metadata=None)
    query_vector_irrelevant = embedder.embed([query_text_irrelevant])[0].vector
    results_irrelevant = store.search(query_vector_irrelevant, top_k=2)
    print(f"Search results for query: '{query_text_irrelevant.content}'")
    if not results_irrelevant:
        print("No results found for the irrelevant query.")
    else:
        for i, result in enumerate(results_irrelevant):
            print(f"RESULT {i+1}:")
            print(f"CONTENT: {result.content}")
            print(f"METADATA: {result.metadata}\n") """

        
if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logger.info(f"EXECUTION TIME: {end_time - start_time:.2f} seconds")
