from pydantic import BaseModel, Field


class IndexResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    chunk_id: int
    source: str
    similarity: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []


class HealthResponse(BaseModel):
    status: str
    indexed_document: str | None = None
