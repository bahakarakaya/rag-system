from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores import FaissVectorStore, BM25Store, HybridRetriever
from rag.generation import OllamaClient, GptClient
from rag.pipeline import IngestionPipeline, QueryPipeline, GenerationPipeline, CrossEncoderReranker

from sentence_transformers import CrossEncoder
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

    embedder = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
    bm25_store = BM25Store(language="english")
    reranker = CrossEncoderReranker(cross_encoder=CrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"))
    chunker = FixedSizeChunker()
    if not Path("data/index_dir/faiss_index.bin").exists():
        store = FaissVectorStore(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")
    else:
        store = FaissVectorStore.load(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")
    retriever = HybridRetriever(vector_store=store, bm25_store=bm25_store, embedder=embedder, reranker=reranker)

    ingestion_pipe = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
        bm25_store=bm25_store
    )
    gen_pipe = GenerationPipeline(
        llm=GptClient(
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        query_pipeline=QueryPipeline(
            retriever=retriever
        ),
        prompt=prompt
    )

    ingestion_pipe.run(source_paths=["data/ai_eng_prj.txt"])

    logger.info("""
    ---------------------------------
    ------------- QUERY -------------
    ---------------------------------
    """)

    results = gen_pipe.run(query=query_text, top_k=3)
    logger.info(f"Question: {query_text}")
    logger.info(f"LLM GENERATED ANSWER: {results['answer']}")
    logger.debug(f"Sources: {results['sources']}")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logger.info(f"EXECUTION TIME: {end_time - start_time:.2f} seconds")
