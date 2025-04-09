from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_retriever(persist_directory: str = "./chroma_db"):
    """Load ChromaDB-based retriever using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever
