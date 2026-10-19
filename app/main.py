from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.ingestion import ingest_file
from app.rag_chain import answer_question_stream
from app.schemas import DocumentInfo, QueryRequest, UploadResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="RAG Document Q&A")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile]):
    ingested = []
    for file in files:
        dest = Path(settings.upload_dir) / file.filename
        dest.write_bytes(await file.read())

        try:
            num_chunks = ingest_file(dest)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        ingested.append(DocumentInfo(filename=file.filename, chunks=num_chunks))

    return UploadResponse(ingested=ingested)


@app.post("/query")
def query(request: QueryRequest):
    return StreamingResponse(
        answer_question_stream(request.question, request.top_k),
        media_type="text/event-stream",
    )
