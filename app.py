import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_all_documents
from chunk import chunk_text

load_dotenv()

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"


@st.cache_resource
def build_index():
    docs = load_all_documents("./docs")
    chunks = chunk_text(docs)

    vectors = []
    metadata = []

    for chunk in chunks:
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk["text"]
        ).data[0].embedding

        vectors.append(emb)
        metadata.append(chunk)

    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    return index, metadata


def search(query, index, metadata, k=3):
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    q_vec = np.array([q_emb]).astype("float32")

    _, idx = index.search(q_vec, k)

    return [metadata[i] for i in idx[0]]


def ask_llm(query, context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
Use only the context below.

Context:
{context}

Question:
{query}
"""

    res = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return res.output_text


st.title("Payment Hub AI Assistant")

query = st.text_input("Ask a question about your documents")

if "index" not in st.session_state:
    with st.spinner("Building index..."):
        st.session_state.index, st.session_state.meta = build_index()

if query:
    with st.spinner("Searching..."):
        results = search(query, st.session_state.index, st.session_state.meta)

    with st.spinner("Generating answer..."):
        answer = ask_llm(query, results)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for r in results:
        st.write(r["doc"])
