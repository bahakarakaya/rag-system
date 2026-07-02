"""Golden-dataset evaluation harness.

Scores this RAG system's retrieval and generation quality against the public
explodinggradients/amnesty_qa dataset (english_v3, eval split, 20 QA pairs).
The dataset's retrieved_contexts passages are ingested as the corpus (into a
dedicated index under data/eval_index_dir/, never the real data/index_dir/);
user_input/reference become the golden questions/ground-truth answers. Our own
GenerationPipeline is run per question, and RAGAS metrics (faithfulness,
answer_relevancy, context_precision, context_recall) judge the results.

Provider: EVAL_LLM_PROVIDER selects the starting provider - "openai" (default),
"gemini", or "ollama" (local, no API key) - for both the generation-under-test LLM
and the RAGAS judge. If the chosen provider hits a quota/rate-limit error, the
*entire* evaluation (generation + judging) is retried from scratch with the next
provider in the chain (openai -> gemini -> ollama, starting wherever configured) -
a run never mixes providers across questions, since that would confound the metrics.

No pass/fail gate - run and read eval_results/run_<timestamp>.json.

Usage:
    python evaluate.py
    EVAL_LLM_PROVIDER=gemini python evaluate.py
"""

from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder

from rag.core.interfaces import Llm
from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores import FaissVectorStore, BM25Store, HybridRetriever
from rag.generation import GptClient, GeminiClient, OllamaClient
from rag.pipeline import IngestionPipeline, QueryPipeline, GenerationPipeline, CrossEncoderReranker, EvaluationPipeline
from utils.langfuse import get_langfuse_client, start_observation

from time import time
from datetime import datetime
from pathlib import Path
import json
import logging
import sys
import os
import requests
from dotenv import load_dotenv

load_dotenv()

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=logging.WARNING,  # suppress third-party library logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

for _ns in ("__main__", "rag", "VectorStore", "utils"):
    logging.getLogger(_ns).setLevel(_log_level)

logger = logging.getLogger(__name__)

DATASET_NAME = "explodinggradients/amnesty_qa"
DATASET_CONFIG = "english_v3"
DATASET_SPLIT = "eval"

CORPUS_DIR = Path("data/golden/amnesty_qa_corpus")
EVAL_INDEX_DIR = Path("data/eval_index_dir")
RESULTS_DIR = Path("eval_results")

TOP_K = 5

PROVIDER_ENV_VAR = "EVAL_LLM_PROVIDER"
PROVIDER_CHAIN = ["openai", "gemini", "ollama"]
API_KEY_ENV_VAR = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}  # ollama needs no key
OLLAMA_URL = "http://localhost:11434"
MODEL_CONFIG = {
    "openai": {
        "generation_model": "gpt-4o-mini",
        "judge_model": "gpt-4o-mini",
        "judge_embedding_model": "text-embedding-3-small",
    },
    "gemini": {
        "generation_model": "gemini-2.5-flash",
        "judge_model": "gemini-2.5-flash",
        "judge_embedding_model": "gemini-embedding-001",
    },
    "ollama": {
        "generation_model": "llama3.1:8b",
        "judge_model": "llama3.1:8b",
        "judge_embedding_model": "nomic-embed-text",
    },
}

PROMPT = """You are a question-answering assistant. Answer the user's question using ONLY the information provided in the context below.

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
    primary_provider = os.getenv(PROVIDER_ENV_VAR, "openai").strip().lower()
    if primary_provider not in PROVIDER_CHAIN:
        raise SystemExit(f"{PROVIDER_ENV_VAR} must be one of {PROVIDER_CHAIN}, got '{primary_provider}'.")

    attempt_order = _provider_attempt_order(primary_provider)
    available_providers = [p for p in attempt_order if _provider_available(p)]
    if not available_providers:
        raise SystemExit(
            "No configured LLM provider is available. Set OPENAI_API_KEY or GEMINI_API_KEY in .env, "
            f"or start a local Ollama server at {OLLAMA_URL} with the {MODEL_CONFIG['ollama']['generation_model']} model pulled."
        )
    skipped = [p for p in attempt_order if p not in available_providers]
    if skipped:
        logger.warning(f"Skipping unavailable provider(s): {skipped}")

    langfuse = get_langfuse_client()

    logger.info(f"Fetching golden dataset '{DATASET_NAME}' ({DATASET_CONFIG}, {DATASET_SPLIT} split)...")
    golden = _load_golden_dataset()

    logger.info("Materializing corpus from golden dataset contexts...")
    corpus_paths = _materialize_corpus(golden)
    logger.info(f"Wrote {len(corpus_paths)} unique corpus passage(s) to {CORPUS_DIR}")

    embedder = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
    bm25_store = BM25Store(language="english")
    reranker = CrossEncoderReranker(cross_encoder=CrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"))
    chunker = FixedSizeChunker()

    EVAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = EVAL_INDEX_DIR / "faiss_index.bin"
    db_path = EVAL_INDEX_DIR / "vector_store_metadata.db"
    if not index_path.exists():
        store = FaissVectorStore(index_path=str(index_path), db_path=str(db_path))
    else:
        store = FaissVectorStore.load(index_path=str(index_path), db_path=str(db_path))

    retriever = HybridRetriever(vector_store=store, bm25_store=bm25_store, embedder=embedder, reranker=reranker)
    ingestion_pipe = IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=store, bm25_store=bm25_store)

    with start_observation(
        langfuse,
        name="evaluation_corpus_ingestion",
        as_type="span",
        input={"sources": corpus_paths, "count": len(corpus_paths)},
    ) as ingestion_span:
        ingestion_pipe.run(source_paths=corpus_paths)
        if ingestion_span is not None:
            ingestion_span.update(output={"ingested_count": len(corpus_paths)})

    # Ingestion is provider-independent, so it only ever runs once. Generation + judging
    # are retried as a single unit with the next available provider on a quota/rate-limit
    # error, so a run never mixes providers across questions. A non-quota error (a real
    # bug or misconfiguration) always surfaces immediately instead of being masked by a retry.
    df = metric_names = provider_used = None
    for i, provider in enumerate(available_providers):
        try:
            df, metric_names, provider_used = _run_full_evaluation(provider, retriever, golden)
            break
        except RuntimeError as e:
            is_last = i == len(available_providers) - 1
            if not _is_quota_or_rate_limit_error(e) or is_last:
                raise
            logger.warning(
                f"'{provider}' run failed with a quota/rate-limit error; "
                f"retrying the entire evaluation with the next available provider..."
            )

    _print_summary(df, metric_names, provider_used)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _write_report(df, metric_names, provider_used, timestamp)
    print(f"Full results written to {out_path}")

    if langfuse is not None:
        langfuse.flush()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _load_golden_dataset():
    """Fetches the amnesty_qa golden QA dataset from the Hugging Face Hub.

    The `datasets` library caches the downloaded files locally, so repeated
    runs after the first are fully offline.
    """
    try:
        return load_dataset(DATASET_NAME, DATASET_CONFIG)[DATASET_SPLIT]
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch the golden dataset '{DATASET_NAME}' from the Hugging Face Hub. "
            "Check network connectivity, or verify a cached copy exists under the HF datasets cache "
            "if running offline."
        ) from e


def _materialize_corpus(golden) -> list[str]:
    """Deduplicates retrieved_contexts passages across all golden rows and writes each
    unique passage to its own .txt file under CORPUS_DIR, so IngestionPipeline tracks and
    chunks each passage independently instead of as one giant concatenated document.
    """
    unique_passages: dict[str, None] = {}
    for row in golden:
        for passage in row["retrieved_contexts"]:
            unique_passages.setdefault(passage, None)

    if not unique_passages:
        raise ValueError("No corpus passages extracted from the golden dataset; check the dataset schema.")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, passage in enumerate(unique_passages):
        path = CORPUS_DIR / f"passage_{i:03d}.txt"
        path.write_text(passage, encoding="utf-8")
        paths.append(str(path))
    return paths


def _provider_attempt_order(primary: str) -> list[str]:
    """Returns the provider try-order starting at `primary`, then the remaining
    providers in PROVIDER_CHAIN order."""
    remaining = [p for p in PROVIDER_CHAIN if p != primary]
    return [primary] + remaining


def _provider_available(provider: str) -> bool:
    """Checks whether a provider is usable: an API key for cloud providers, or a
    reachable local server for Ollama."""
    if provider in API_KEY_ENV_VAR:
        return bool(os.getenv(API_KEY_ENV_VAR[provider]))
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def _build_generation_llm(provider: str) -> Llm:
    """Builds the generation-under-test LLM for the given provider."""
    cfg = MODEL_CONFIG[provider]
    if provider == "openai":
        return GptClient(model_name=cfg["generation_model"], api_key=os.getenv(API_KEY_ENV_VAR[provider]))
    if provider == "gemini":
        return GeminiClient(model_name=cfg["generation_model"], api_key=os.getenv(API_KEY_ENV_VAR[provider]))
    return OllamaClient(model_name=cfg["generation_model"], ollama_url=OLLAMA_URL)


def _build_judge(provider: str):
    """Builds the RAGAS-compatible judge LLM and embeddings for the given provider."""
    cfg = MODEL_CONFIG[provider]
    if provider == "openai":
        api_key = os.getenv(API_KEY_ENV_VAR[provider])
        judge_llm = LangchainLLMWrapper(ChatOpenAI(model=cfg["judge_model"], api_key=api_key, temperature=0))
        judge_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=cfg["judge_embedding_model"], api_key=api_key)
        )
    elif provider == "gemini":
        api_key = os.getenv(API_KEY_ENV_VAR[provider])
        judge_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model=cfg["judge_model"], google_api_key=api_key, temperature=0)
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model=cfg["judge_embedding_model"], google_api_key=api_key)
        )
    else:
        judge_llm = LangchainLLMWrapper(ChatOllama(model=cfg["judge_model"], base_url=OLLAMA_URL, temperature=0))
        judge_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=cfg["judge_embedding_model"], base_url=OLLAMA_URL)
        )
    return judge_llm, judge_embeddings


def _run_generation_for_golden_set(gen_pipe: GenerationPipeline, golden) -> list[SingleTurnSample]:
    """Runs our own GenerationPipeline for each golden question and assembles RAGAS samples.

    Uses our own retrieved chunks (not the dataset's pre-baked retrieved_contexts) so that
    context_precision/context_recall actually measure this system's retrieval quality.
    """
    samples = []
    for row in golden:
        result = gen_pipe.run(query=row["user_input"], top_k=TOP_K)
        retrieved_contexts = [scored_chunk.chunk.content for scored_chunk in result["chunks"]]
        samples.append(SingleTurnSample(
            user_input=row["user_input"],
            response=result["answer"],
            retrieved_contexts=retrieved_contexts,
            reference=row["reference"],
        ))
        logger.info(f"Generated answer for: {row['user_input'][:60]}...")
    return samples


def _run_full_evaluation(provider: str, retriever: HybridRetriever, golden):
    """Runs generation for all golden questions and RAGAS evaluation end-to-end using the
    given provider for both the generation-under-test LLM and the RAGAS judge.

    Raises:
        RuntimeError: If generation or judging fails (chains the original provider error
            so the caller can inspect it and decide whether to retry with a fallback provider).
    """
    logger.info(f"Running full evaluation with provider='{provider}'...")
    try:
        gen_pipe = GenerationPipeline(
            llm=_build_generation_llm(provider),
            query_pipeline=QueryPipeline(retriever=retriever),
            prompt=PROMPT,
        )
        samples = _run_generation_for_golden_set(gen_pipe, golden)
    except Exception as e:
        raise RuntimeError(f"Generation failed for provider '{provider}': {str(e)}") from e

    eval_dataset = EvaluationDataset(samples=samples)
    judge_llm, judge_embeddings = _build_judge(provider)
    eval_pipeline = EvaluationPipeline(judge_llm=judge_llm, judge_embeddings=judge_embeddings)
    results = eval_pipeline.run(eval_dataset)
    df = results.to_pandas()
    metric_names = [m.name for m in eval_pipeline.metrics]
    return df, metric_names, provider


def _is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    """Walks the exception cause chain looking for a provider quota/rate-limit signal.

    Only a genuine quota/rate-limit error should trigger a whole-run retry with the
    fallback provider - any other failure (a real bug) should surface immediately
    instead of being silently masked by a retry.
    """
    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None)
        if status_code is None:
            status_code = getattr(current, "code", None)
        if status_code == 429:
            return True
        type_name = type(current).__name__.lower()
        if "ratelimit" in type_name or "resourceexhausted" in type_name or "quotaexceeded" in type_name:
            return True
        if "insufficient_quota" in str(current).lower():
            return True
        current = current.__cause__
    return False


def _print_summary(df, metric_names: list[str], provider: str) -> None:
    """Prints the mean score per RAGAS metric to the console."""
    print(f"\n=== RAGAS Evaluation Summary ({DATASET_NAME}, {len(df)} questions, provider={provider}) ===")
    for name in metric_names:
        print(f"{name:20s}: {df[name].mean():.3f}")
    print()


def _write_report(df, metric_names: list[str], provider: str, timestamp: str) -> Path:
    """Writes full per-question scores and aggregate means to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    cfg = MODEL_CONFIG[provider]
    payload = {
        "timestamp": timestamp,
        "dataset": f"{DATASET_NAME} ({DATASET_CONFIG}, {DATASET_SPLIT} split)",
        "row_count": len(df),
        "provider": provider,
        "judge_model": cfg["judge_model"],
        "generation_model": cfg["generation_model"],
        "top_k": TOP_K,
        "aggregate_scores": {name: float(df[name].mean()) for name in metric_names},
        "per_question": df.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logger.info(f"EXECUTION TIME: {end_time - start_time:.2f} seconds")
