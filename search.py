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


def build_index():
    docs = load_all_documents("./docs")
    chunks = chunk_text(docs)

    vectors = []
    metadata = []

    for chunk in chunks:
        emb = get_embedding(chunk["text"])
        vectors.append(emb)
        metadata.append(chunk)

    vectors = np.array(vectors).astype("float32")

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, metadata


def search(query, index, metadata, k=3):
    query_vec = np.array([get_embedding(query)]).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        results.append(metadata[i])

    return results


def ask_llm(query, context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are a Payment Hub documentation assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":
    print("Building index...")
    index, metadata = build_index()

    print("Ready for queries")

    while True:
        q = input("\nAsk: ")

        if q.lower() in ["exit", "quit"]:
            break

        results = search(q, index, metadata)

        answer = ask_llm(q, results)

        print("\nANSWER:\n", answer)