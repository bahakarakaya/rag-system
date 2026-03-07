from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores.faiss import FaissVectorStore
from rag.generation import OllamaClient, GptClient
from rag.pipeline import IngestionPipeline, QueryPipeline, GenerationPipeline
from rag.core.models import Chunk

from datasets import load_dataset
from ragas import EvaluationDataset
from utils.hashing import compute_content_hash
from time import time
from pathlib import Path
import logging
import sys
import os
from dotenv import load_dotenv


load_dotenv()

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


prompt = """You are a question-answering assistant. Answer the user's question using ONLY the information provided in the context below.

Rules:
- Base your answer strictly on the provided context. Do not use prior knowledge or make assumptions beyond what is stated.
- If the context does not contain sufficient information to answer the question, respond with exactly: "I don't have enough information in the provided context to answer this question." Nothing more.
- Do not speculate or hallucinate details that are not in the context.
- Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""


def main():

    query_text = "What are the main components of a RAG system and how do they interact with each other?"
    """ query_text_irrelevant = "What is the weather like in Antalya today?"
    query_text_irrelevant = "How to start an F-16 fighter jet?"
    query_text = query_text_irrelevant """

    logger.info("RAG system starting...")

    logger.info("""
    ---------------------------------
    ----------- INGESTION -----------
    ---------------------------------
    """)

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
    gen_pipe = GenerationPipeline(
        llm=GptClient(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        query_pipeline=QueryPipeline(
            embedder=embedder,
            store=store
        ),
        prompt=prompt
    )

    ingestion_pipe.run(source_paths=["data/ai_eng_prj.txt"])

    logger.info("""
    ---------------------------------
    ------------- QUERY -------------
    ---------------------------------
    """)

    results = gen_pipe.run(question=query_text, top_k=3)
    logger.info(f"Question: {query_text}")
    logger.info(f"LLM GENERATED ANSWER: {results['answer'] if gen_pipe.llm.__class__ == OllamaClient else results}")

    if not results:
        logger.info("No results found for the relevant query. Returning empty list.")
    else:
        logger.debug(f"Search results for query: '{query_text}'")
        for i, result in enumerate(results):
            logger.debug(f"RESULT {i+1}:")
            logger.debug(f"CONTENT: {result.content.strip()}")
            logger.debug(f"METADATA: {result.metadata}\n")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logger.info(f"EXECUTION TIME: {end_time - start_time:.2f} seconds")
