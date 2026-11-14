from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """Shared state object passed between nodes in the agentic workflow."""
    question: str
    generation: str
    documents: List[Document]
    web_search_needed: bool
    retry_count: int
    route: Optional[str]
