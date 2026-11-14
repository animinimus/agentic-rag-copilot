from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return the local HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ensure_index_exists(pc: Pinecone, index_name: str, dimension: int = 384) -> None:
    """Create the Pinecone serverless index if it does not exist with the expected dimension."""
    if pc.has_index(index_name):
        existing_dim = pc.describe_index(index_name).dimension
        if existing_dim == dimension:
            return
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def get_vectorstore() -> PineconeVectorStore:
    """Connect to the Pinecone index and return a vector store interface."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index_exists(pc, settings.pinecone_index_name)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


def index_documents(documents: List[Document]) -> List[str]:
    """Embed and upsert document chunks into the Pinecone index."""
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(documents)


def get_retriever() -> VectorStoreRetriever:
    """Return a retriever instance configured for top-k similarity search."""
    settings = get_settings()
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k},
    )
