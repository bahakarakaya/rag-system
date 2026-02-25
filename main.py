from VectorStore.chunkers import FixedSizeChunker
from VectorStore.loaders import TextFileLoader
from VectorStore.embedders import SentenceTransformersEmbedder
from VectorStore.stores.faiss import FaissVectorStore
from rag.core.models import Chunk
from utils.hashing import compute_content_hash
from time import time
from pathlib import Path



def main():
    query_text = "What are the main components of a RAG system and how do they interact with each other?"
    query_text_irrelevant = "How is the weather like in Izmir today?"

    print("Hello from rag-system!")

    # TEST
    loader = TextFileLoader("ai_eng_prj.txt")
    embedder = SentenceTransformersEmbedder()
    chunker = FixedSizeChunker()
    if not Path("data/index_dir/faiss_index.bin").exists():
        store = FaissVectorStore(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")
    else:
        store = FaissVectorStore.load(index_path="data/index_dir/faiss_index.bin", db_path="data/index_dir/vector_store_metadata.db")

    documents = list(loader.load())
    for doc in documents:
        source = doc.metadata["source"]
        current_hash = compute_content_hash(doc.content)
        is_indexed = store.db_manager.is_document_indexed(source)
        stored_hash = store.db_manager.get_stored_hash(source)

        if is_indexed and stored_hash == current_hash:
            print(f"Document '{source}' is already indexed and unchanged, skipping.")
            continue

        if is_indexed:
            print(f"Document '{source}' has changed, re-indexing...")
            stale_ids = store.db_manager.delete_by_source(source)
            store.remove_vectors(stale_ids)
        else:
            print(f"New document '{source}', indexing...")

        doc.metadata["content_hash"] = current_hash
        chunks = chunker.chunk(doc)
        print(f"Document: {doc.metadata['doc_id']} - {len(chunks)} chunks created.")
        for chunk in chunks:
            print(f"-------------------- CHUNK -------------------- \n {chunk.metadata}: {chunk.content}")

        print(f"Embedding chunks for document '{doc.metadata['doc_id']}'...")
        embedded_chunks = embedder.embed(chunks)
        for embedded_chunk in embedded_chunks:
            print(f"Embedded Chunk Metadata: {embedded_chunk.chunk.metadata}")
            print(f"Embedded Chunk Vector (first 3 dimensions): {embedded_chunk.vector[:3]}\n")

        store.save(embedded_chunks)
    
    print("\nPerforming search with relevant query...")
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
            print(f"METADATA: {result.metadata}\n")

        
if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"\n     EXECUTION TIME: {end_time - start_time:.2f} seconds")
