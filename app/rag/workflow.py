import os
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.vectorstore import get_retriever


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""

    relevant: bool = Field(
        description="True if the document is relevant to the user question, False otherwise."
    )


def get_llm():
    """Initialize the chat model from application settings."""
    settings = get_settings()
    if settings.openai_api_key:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    if settings.groq_api_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0,
        )
    raise ValueError("Missing LLM API key: configure OPENAI_API_KEY or GROQ_API_KEY.")


def retrieve(state: AgentState) -> Dict[str, Any]:
    """Retrieve top-k relevant documents from the vector store."""
    retriever = get_retriever()
    documents = retriever.invoke(state["question"])
    return {"documents": documents}


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Filter retrieved documents by relevance and flag if web fallback is needed."""
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeDocuments)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether a retrieved document helps answer an IT support question. "
                "Respond with relevant=True if the document contains keywords or semantic meaning related to the question.",
            ),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )
    grader_chain = prompt | structured_grader

    filtered_docs: List[Document] = []
    for doc in state["documents"]:
        score = grader_chain.invoke(
            {"question": state["question"], "document": doc.page_content}
        )
        if score.relevant:
            filtered_docs.append(doc)

    web_search_needed = len(filtered_docs) == 0
    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed,
    }


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    """Rewrite the user question into a concise query optimized for web search."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query rewriter. Convert the user's IT support question into a concise, "
                "keyword-focused web search query while preserving the exact technical intent.",
            ),
            ("human", "Original question: {question}"),
        ]
    )
    rewriter = prompt | llm | StrOutputParser()
    better_query = rewriter.invoke({"question": state["question"]}).strip()
    return {
        "question": better_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def web_search(state: AgentState) -> Dict[str, Any]:
    """Run a Tavily web search and normalize results into Document objects."""
    settings = get_settings()
    if settings.tavily_api_key:
        os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
    elif not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("Missing Tavily API key: configure TAVILY_API_KEY.")

    search_tool = TavilySearch(max_results=3)
    response = search_tool.invoke({"query": state["question"]})

    results = response.get("results", []) if isinstance(response, dict) else response
    web_docs: List[Document] = []
    for item in results:
        if isinstance(item, dict):
            web_docs.append(
                Document(
                    page_content=item.get("content", ""),
                    metadata={"source": item.get("url", "web_search")},
                )
            )
    return {"documents": web_docs}


def generate(state: AgentState) -> Dict[str, Any]:
    """Generate a grounded IT support response from retrieved or web documents."""
    llm = get_llm()
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state["documents"]
    )
    source_note = (
        "Note: Answer is based on external web search results rather than internal IT documentation."
        if state.get("web_search_needed")
        else "Answer is grounded in internal IT documentation."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an enterprise IT support copilot. Answer the user's question clearly using "
                "only the provided context, and cite the sources. Never ask for passwords or MFA codes.\n"
                "{source_note}",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "source_note": source_note,
        }
    )
    return {"generation": answer}


def decide_to_generate(state: AgentState) -> str:
    """Route to query rewriting for web fallback or directly to answer generation."""
    if state.get("web_search_needed"):
        return "rewrite_query"
    return "generate"


def build_workflow():
    """Compile and return the LangGraph agentic RAG state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

