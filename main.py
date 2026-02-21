def main():
    print("Hello from rag-system!")

    from VectorStore.chunkers import FixedSizeChunker
    from VectorStore.loaders import TextFileLoader

    # TEST
    loader = TextFileLoader("ai_eng_prj.txt")
    chunker = FixedSizeChunker()

    documents = list(loader.load())
    for doc in documents:
        chunks = chunker.chunk_by_characters(doc)
        print(f"Document: {doc.metadata['doc_id']} - {len(chunks)} chunks created.")
        for chunk in chunks:
            print(f"-------------------- CHUNK -------------------- \n {chunk.metadata}: {chunk.content}")
if __name__ == "__main__":
    main()
