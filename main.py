def main():
    print("Hello from rag-system!")

    from VectorStore.chunkers import FixedSizeChunker
    from VectorStore.loaders import TextFileLoader
    from VectorStore.embedders import SentenceTransformersEmbedder

    # TEST
    loader = TextFileLoader("ai_eng_prj.txt")
    chunker = FixedSizeChunker()

    documents = list(loader.load())
    for doc in documents:
        chunks = chunker.chunk_by_characters(doc)
        print(f"Document: {doc.metadata['doc_id']} - {len(chunks)} chunks created.")
        for chunk in chunks:
            print(f"-------------------- CHUNK -------------------- \n {chunk.metadata}: {chunk.content}")
        
        print(f"Embedding chunks for document '{doc.metadata['doc_id']}'...")
        embedder = SentenceTransformersEmbedder()
        embedded_chunks = embedder.embed(chunks)
        for embedded_chunk in embedded_chunks:
            print(f"Embedded Chunk Metadata: {embedded_chunk.chunk.metadata}")
            print(f"Embedded Chunk Vector (first 3 dimensions): {embedded_chunk.vector[:3]}\n")

        
if __name__ == "__main__":
    main()
