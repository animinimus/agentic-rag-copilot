from pathlib import Path
from typing import List
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.vectorstore import index_documents
from app.services.ingestion import chunk_document, load_text_file


def ingest_sample_kb() -> int:
    """Load, chunk, and index all sample knowledge base files into Pinecone."""
    settings = get_settings()
    kb_dir = Path(settings.sample_kb_dir)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Sample KB directory not found: {kb_dir}")

    all_docs: List[Document] = []
    for file_path in sorted(kb_dir.iterdir()):
        if file_path.suffix.lower() in {".md", ".txt"}:
            raw_text = load_text_file(file_path)
            chunks = chunk_document(raw_text, source=file_path.name)
            all_docs.extend(chunks)

    if not all_docs:
        print(f"No supported documents found in {kb_dir}")
        return 0

    ids = index_documents(all_docs)
    print(f"Indexed {len(ids)} chunks from {kb_dir} into '{settings.pinecone_index_name}'.")
    return len(ids)


if __name__ == "__main__":
    ingest_sample_kb()
