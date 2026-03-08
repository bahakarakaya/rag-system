import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

from rag.ingestion.chunkers import FixedSizeChunker
from rag.ingestion.embedders import SentenceTransformersEmbedder
from rag.stores import FaissVectorStore, BM25Store, HybridRetriever
from rag.generation import GptClient
from rag.pipeline import (
    IngestionPipeline,
    QueryPipeline,
    GenerationPipeline,
    CrossEncoderReranker,
)
from rag.core.models import Chunk, ChunkMetadata
from sentence_transformers import CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-distilroberta-v1": 768,
}

CROSS_ENCODER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
]

GPT_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]

PROMPT_TEMPLATE = """\
You are a question-answering assistant. Answer the user's question using ONLY \
the information provided in the context below.

Rules:
- Base your answer strictly on the provided context. Do not use prior knowledge \
or make assumptions beyond what is stated.
- If the context does not contain sufficient information to answer the question, \
respond with exactly: "I don't have enough information in the provided context \
to answer this question." Nothing more.
- Do not speculate or hallucinate details that are not in the context.
- Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG System", page_icon="🔍", layout="wide")

st.title("🔍 RAG System")
st.markdown(
    "<p style='color:gray; margin-top:-10px;'>"
    "Hybrid Search ⇒ FAISS Vector + BM25 retrieval → Cross-encoder reranking → LLM answer generation"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – ⚙️ Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Language Model ─────────────────────────────────────────────────
    st.subheader("🤖 Language Model")
    gpt_model = st.selectbox("GPT Model", GPT_MODELS)
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Leave blank to use API_KEY from .env",
    )

    # ── Embeddings ─────────────────────────────────────────────────────
    st.subheader("🔢 Embeddings")
    embedding_model = st.selectbox(
        "Sentence-Transformer Model", list(EMBEDDING_MODELS.keys())
    )
    cross_encoder_model = st.selectbox(
        "Cross-Encoder (Reranker)", CROSS_ENCODER_MODELS
    )

    # ── Chunking ───────────────────────────────────────────────────────
    st.subheader("✂️ Chunking")
    chunk_size = st.slider("Chunk Size (characters)", 100, 2000, 500, step=50)
    overlap = st.slider("Overlap (characters)", 0, 500, 75, step=25)

    # ── Retrieval ──────────────────────────────────────────────────────
    st.subheader("🔍 Retrieval")
    top_k = st.slider("Top-K Results", 1, 20, 5)


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedder(model_name: str) -> SentenceTransformersEmbedder:
    """Load the sentence-transformer embedder (heavy – cached)."""
    return SentenceTransformersEmbedder(model_name=model_name)


@st.cache_resource
def get_reranker(model_name: str) -> CrossEncoderReranker:
    """Load the cross-encoder reranker (heavy – cached)."""
    return CrossEncoderReranker(cross_encoder=CrossEncoder(model_name=model_name))


@st.cache_resource
def get_stores(
    _embedding_model: str,
) -> tuple[FaissVectorStore, BM25Store]:
    """Initialise FAISS vector-store and BM25 store, warming BM25 from the
    persisted SQLite DB so that queries work immediately on app restart."""
    dimension = EMBEDDING_MODELS[_embedding_model]
    safe_name = _embedding_model.replace("/", "_")
    index_dir = Path("data/index_dir") / safe_name
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = str(index_dir / "faiss_index.bin")
    db_path = str(index_dir / "vector_store_metadata.db")

    if Path(index_path).exists():
        vector_store = FaissVectorStore.load(index_path=index_path, db_path=db_path)
    else:
        vector_store = FaissVectorStore(
            index_path=index_path, db_path=db_path, dimension=dimension
        )

    bm25_store = BM25Store(language="english")

    # Warm BM25 from persisted chunks in DB
    rows = vector_store.db_manager.get_all_chunks()
    if rows:
        chunks = [
            Chunk(
                content=row["content"],
                metadata=ChunkMetadata(
                    source=row["source"],
                    doc_id=row["doc_id"],
                    format=row["format"],
                    chunk_index=row["chunk_index"],
                    start_index=row["start_index"],
                    end_index=row["end_index"],
                    section=row.get("section"),
                    page=row.get("page"),
                    created_at=row.get("created_at"),
                ),
            )
            for row in rows
        ]
        bm25_store.save(chunks)

    return vector_store, bm25_store


# ---------------------------------------------------------------------------
# Build pipeline components from current sidebar state
# ---------------------------------------------------------------------------
embedder = get_embedder(embedding_model)
reranker = get_reranker(cross_encoder_model)
vector_store, bm25_store = get_stores(embedding_model)

effective_key = api_key_input if api_key_input else os.getenv("OPENAI_API_KEY")
if not effective_key:
    st.sidebar.warning("⚠️ No OpenAI API key provided. Set one above or in your .env file.")
llm = GptClient(model_name=gpt_model, api_key=effective_key)

retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_store=bm25_store,
    embedder=embedder,
    reranker=reranker,
)
query_pipeline = QueryPipeline(retriever=retriever)
generation_pipeline = GenerationPipeline(
    llm=llm, query_pipeline=query_pipeline, prompt=PROMPT_TEMPLATE
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_ingested" not in st.session_state:
    # Consider documents already ingested if the FAISS index has vectors
    st.session_state.documents_ingested = vector_store.index.ntotal > 0
if "ingested_file_hashes" not in st.session_state:
    st.session_state.ingested_file_hashes: set[str] = set()

# ---------------------------------------------------------------------------
# Helper – render retrieved chunks grouped by source file
# ---------------------------------------------------------------------------
def _render_sources(scored_chunks: list) -> None:
    """Render retrieved chunks grouped by source filename."""
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for sc in scored_chunks:
        groups[sc.chunk.metadata.source].append(sc)
    for source, chunks in groups.items():
        st.markdown(f"**`{Path(source).name}`**")
        for sc in chunks:
            st.markdown(
                f"<div style='border-left:3px solid #4a90d9;"
                f"padding:8px 12px;margin:4px 0;border-radius:4px;font-size:0.85em;"
                f"opacity:0.9;'>"
                f"{sc.chunk.content}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
chat_tab, about_tab = st.tabs(["💬 Chat", "ℹ️ About"])

# ===========================  CHAT TAB  ====================================
with chat_tab:
    # ── Document ingestion ─────────────────────────────────────────────
    st.subheader("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload text documents", type=["txt"], accept_multiple_files=True
    )

    if uploaded_files:
        import hashlib
        new_files = [
            f for f in uploaded_files
            if hashlib.md5(f.getvalue()).hexdigest() not in st.session_state.ingested_file_hashes
        ]
        if new_files:
            with st.spinner(f"Ingesting {len(new_files)} document(s)…"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = []
                    for f in new_files:
                        dest = Path(tmpdir) / f.name
                        dest.write_bytes(f.getvalue())
                        paths.append(str(dest))
                    try:
                        chunker = FixedSizeChunker(
                            chunk_size=chunk_size, overlap=overlap
                        )
                        ingestion_pipeline = IngestionPipeline(
                            chunker=chunker,
                            embedder=embedder,
                            vector_store=vector_store,
                            bm25_store=bm25_store,
                        )
                        ingestion_pipeline.run(source_paths=paths)
                        for f in new_files:
                            st.session_state.ingested_file_hashes.add(
                                hashlib.md5(f.getvalue()).hexdigest()
                            )
                        st.session_state.documents_ingested = True
                        st.session_state._ingest_success = len(new_files)
                        st.rerun()
                    except Exception as e:
                        logger.exception("Ingestion failed")
                        st.error(f"❌ Ingestion failed: {e}")

    if "_ingest_success" in st.session_state:
        count = st.session_state["_ingest_success"]
        del st.session_state["_ingest_success"]
        st.success(f"✅ Successfully ingested {count} document(s)!")

    st.divider()

    # ── Chat interface ─────────────────────────────────────────────────
    if not st.session_state.documents_ingested:
        st.info("📤 Upload and ingest documents to start chatting.")

    # Render conversation history in a scrollable container
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("chunks"):
                    with st.expander("📄 Retrieved Sources"):
                        _render_sources(msg["chunks"])

    # Accept user input
    if prompt_input := st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state.documents_ingested,
    ):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_input)

        # Generate assistant response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        result = generation_pipeline.run(
                            query=prompt_input, top_k=top_k
                        )
                        answer = result["answer"]
                        retrieved_chunks = result.get("chunks", [])
                    except Exception as e:
                        logger.exception("Generation failed")
                        answer = f"⚠️ An error occurred while generating the answer: {e}"
                        retrieved_chunks = []

                st.markdown(answer)
                if retrieved_chunks:
                    with st.expander("📄 Retrieved Sources"):
                        _render_sources(retrieved_chunks)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "chunks": retrieved_chunks}
                )

    # Scroll the chat container to the bottom after each response
    st.components.v1.html(
        """
        <script>
            const containers = window.parent.document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
            if (containers.length) {
                const chat = containers[containers.length - 1];
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
        """,
        height=0,
    )


# ===========================  ABOUT TAB  ===================================
with about_tab:
    st.header("About this RAG System")
    st.markdown(
        "This application implements a **Retrieval-Augmented Generation (RAG)** "
        "pipeline with a hybrid retrieval strategy."
    )

    st.subheader("Architecture")
    st.code(
        """\
Documents
   │
   ▼
TextFileLoader ──► FixedSizeChunker ──► SentenceTransformers
                                                     │
                             ┌───────────────────────┴────────────────────────┐
                             ▼                                                ▼
                       FAISS VectorStore                                 BM25 Store
                             │                                                │
                             └──────────────────────┐  ┌──────────────────────┘
                                                    ▼  ▼
                                           Reciprocal Rank Fusion
                                                     │
                                                     ▼
                                            Cross-Encoder Reranker
                                                     │
                                                     ▼
                                               Top-K Chunks
                                                     │
                                                     ▼
                                             LLM (GPT / Ollama)
                                                     │
                                                     ▼
                                                   Answer""",
        language=None,
    )

    st.subheader("Components")
    st.table(
        {
            "Component": [
                "Embedder",
                "Vector Store",
                "Keyword Store",
                "Fusion",
                "Reranker",
                "LLM",
            ],
            "Default": [
                "all-MiniLM-L6-v2",
                "FAISS (cosine similarity)",
                "BM25 (rank-bm25)",
                "Reciprocal Rank Fusion (RRF)",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "OpenAI GPT",
            ],
        },
        width="content"
    )

    st.subheader("How to Use")
    st.markdown(
        """\
1. **Configure** the system using the **⚙️ Configuration** panel on the left sidebar \
— choose your LLM provider, embedding model, chunking parameters, and retrieval settings.
2. **Upload** one or more `.txt` documents in the **📄 Document Ingestion** section of \
the Chat tab.
3. **Click** the **🚀 Ingest Documents** button and wait for the ingestion to complete.
4. **Ask questions** in the chat input box — the system will retrieve the most relevant \
chunks and generate an answer using the selected LLM.
5. **Review sources** — expand the *📄 Retrieved Sources* section below each answer to \
see which document chunks were used.
"""
    )
