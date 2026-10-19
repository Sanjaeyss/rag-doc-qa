from pydantic import BaseModel


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    ingested: list[DocumentInfo]


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    source: str
    cited_text: str
