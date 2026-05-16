from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Public metadata returned for an indexed document."""

    document_id: str
    filename: str
    title: str
    document_kind: str = "unstructured"
    page_count: int | None = None
    record_count: int | None = None
    chunk_count: int
    status: str = "ready"


class AskQuestionRequest(BaseModel):
    """Request body for answering a question against an indexed document."""

    question: str = Field(..., min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=8)


class RetrievedSection(BaseModel):
    """Document chunk used as source context for a generated answer."""

    text: str
    distance: float
    metadata: dict


class AskQuestionResponse(BaseModel):
    """Generated answer plus source sections used by the LLM."""

    answer: str
    document_id: str
    question: str
    retrieved_sections: list[RetrievedSection]
