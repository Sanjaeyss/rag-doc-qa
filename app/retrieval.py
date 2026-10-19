from app.config import settings
from app.ingestion import get_vectorstore


def retrieve(question: str, k: int | None = None) -> list:
    """Semantic similarity search over the ingested document chunks."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(question, k=k or settings.top_k)
