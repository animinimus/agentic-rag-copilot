from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )


def get_vectorstore() -> PineconeVectorStore:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


def index_documents(documents: List[Document]) -> List[str]:
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(documents)


def get_retriever() -> VectorStoreRetriever:
    settings = get_settings()
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k},
    )
