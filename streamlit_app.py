import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

st.title("Fintech AI Document Search")

# Load vector DB
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

query = st.text_input("Ask a question")

if query:
    results = db.similarity_search(query, k=5)

    st.subheader("Top Results")

    for i, doc in enumerate(results):
        st.markdown(f"### Result {i+1}")
        st.write("Source:", doc.metadata.get("source"))
        st.write("Chunk:", doc.page_content[:500])
        st.write("---")