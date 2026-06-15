import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_all_documents
from chunk import chunk_text

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def build_chunks():
    docs = load_all_documents("./docs")
    chunks = chunk_text(docs)
    return chunks


def build_faiss_index(chunks):
    vectors = []
    metadata = []

    print("Generating embeddings...")

    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk["text"])
        vectors.append(emb)
        metadata.append(chunk)

        if i % 10 == 0:
            print(f"Processed {i}/{len(chunks)} chunks")

    vectors = np.array(vectors).astype("float32")

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, metadata


if __name__ == "__main__":
    chunks = build_chunks()

    print("Total chunks:", len(chunks))

    index, metadata = build_faiss_index(chunks)

    print("FAISS index built")
    print("Vectors stored:", index.ntotal)