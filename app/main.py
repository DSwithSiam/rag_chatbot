import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.constants import NO_INFO_MESSAGE
from app.document_loader import validate_extension
from app.rag_service import RAGService
from app.schemas import ChatRequest, ChatResponse, Citation, HealthResponse, IndexResponse


log_path = Path(settings.log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            filename=log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
service = RAGService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_document=service.indexed_document)


@app.post("/v1/documents/index", response_model=IndexResponse)
async def index_document(file: UploadFile = File(...)) -> IndexResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    try:
        validate_extension(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(file.filename).suffix.lower()
    safe_name = f"{uuid4().hex}{suffix}"
    destination = Path(settings.documents_dir) / safe_name

    content = await file.read()
    destination.write_bytes(content)

    try:
        chunk_count = service.index_document(file_path=destination, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    logger.info("Indexed document '%s' into %s chunks", file.filename, chunk_count)
    return IndexResponse(
        document_id=safe_name,
        filename=file.filename,
        chunks=chunk_count,
        message="Document indexed successfully.",
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not service.indexed_document:
        raise HTTPException(status_code=400, detail="No document indexed yet.")

    try:
        answer, retrieved = service.answer_question(
            session_id=payload.session_id,
            question=payload.question,
        )
    except Exception as exc:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}") from exc

    if answer == NO_INFO_MESSAGE:
        return ChatResponse(session_id=payload.session_id, answer=NO_INFO_MESSAGE, citations=[])

    citations = [
        Citation(chunk_id=item.chunk_id, source=item.source, similarity=round(item.similarity, 4))
        for item in retrieved
    ]
    logger.info("Answered session_id=%s with %d citations", payload.session_id, len(citations))
    return ChatResponse(session_id=payload.session_id, answer=answer, citations=citations)
