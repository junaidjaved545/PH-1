from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import load_all_documents


def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for doc_name, text in documents.items():
        split_chunks = splitter.split_text(text)

        for i, chunk in enumerate(split_chunks):
            chunks.append({
                "doc": doc_name,
                "chunk_id": i,
                "text": chunk
            })

    return chunks


if __name__ == "__main__":
    docs = load_all_documents("./docs")

    print("Docs loaded:", len(docs))

    chunks = chunk_text(docs)

    print("Total Chunks Created:", len(chunks))
    print("\nSample Chunk:\n", chunks[0])