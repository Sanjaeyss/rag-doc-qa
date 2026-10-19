from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore


def load_document(path: Path) -> list:
    loader_cls = _LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader_cls(str(path)).load()


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_file(path: Path) -> int:
    """Load, chunk, embed, and store a single file. Returns the chunk count."""
    docs = load_document(path)
    for doc in docs:
        doc.metadata["source"] = path.name

    chunks = chunk_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    get_vectorstore().add_documents(chunks)
    return len(chunks)
