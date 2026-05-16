import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.models import AskQuestionRequest, AskQuestionResponse, DocumentResponse
from backend.services.document_service import (
    DocumentNotFoundError,
    DocumentProcessingError,
    ask_question,
    delete_document,
    list_documents,
    upload_document,
)
from utils.llm_answer import MissingOpenAIKeyError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Document Chatbot API",
    description=(
        "FastAPI backend for structured and unstructured document upload, "
        "retrieval, and RAG answers."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check used by the Streamlit frontend."""
    return {"status": "ok"}


@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document_endpoint(file: UploadFile = File(...)) -> DocumentResponse:
    """Upload a document, extract text, chunk it, embed it, and store it."""
    allowed_types = {
        "application/pdf",
        "application/json",
        "text/plain",
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF, TXT, CSV, JSON, or Excel file.",
        )

    try:
        return upload_document(file)
    except DocumentProcessingError as error:
        logger.warning("Document processing failed: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected upload failure")
        raise HTTPException(
            status_code=500,
            detail="The document could not be uploaded and indexed.",
        ) from error


@app.post("/ask-question", response_model=AskQuestionResponse)
def ask_question_endpoint(payload: AskQuestionRequest) -> AskQuestionResponse:
    """Answer a user question using retrieved chunks from an indexed document."""
    try:
        return ask_question(
            question=payload.question,
            document_id=payload.document_id,
            top_k=payload.top_k,
        )
    except MissingOpenAIKeyError as error:
        raise HTTPException(
            status_code=401,
            detail="Missing OpenAI API key. Add OPENAI_API_KEY to .env.",
        ) from error
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected question-answering failure")
        raise HTTPException(
            status_code=500,
            detail="The question could not be answered.",
        ) from error


@app.get("/documents", response_model=list[DocumentResponse])
def documents_endpoint() -> list[DocumentResponse]:
    """Return metadata for indexed documents."""
    return list_documents()


@app.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: str) -> dict[str, str]:
    """Delete an indexed document and its vector collection."""
    try:
        delete_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected delete failure")
        raise HTTPException(
            status_code=500,
            detail="The document could not be deleted.",
        ) from error

    return {"status": "deleted", "document_id": document_id}
