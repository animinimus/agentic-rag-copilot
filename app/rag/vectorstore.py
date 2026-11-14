from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Initialize and return the OpenAI embeddings model based on app settings."""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )


def get_vectorstore() -> PineconeVectorStore:
    """Connect to the Pinecone index and return a vector store interface."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
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
