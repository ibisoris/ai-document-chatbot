import json
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.models import AskQuestionResponse, DocumentResponse, RetrievedSection
from utils.embeddings import create_embeddings, load_embedding_model
from utils.llm_answer import generate_answer
from utils.document_reader import DocumentReadError, read_document
from utils.retriever import (
    create_or_replace_collection,
    delete_collection,
    search_similar_chunks,
)
from utils.text_chunker import split_text_into_chunks


logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("documents")
REGISTRY_PATH = Path("vector_store") / "documents.json"
COLLECTION_PREFIX = "document"


class DocumentNotFoundError(Exception):
    """Raised when an indexed document cannot be found."""


class DocumentProcessingError(Exception):
    """Raised when an uploaded document cannot be processed for RAG."""


def ensure_storage() -> None:
    """Create directories used by document and vector storage."""
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.parent.mkdir(exist_ok=True)


def load_registry() -> dict:
    """Load indexed document metadata from disk."""
    ensure_storage()

    if not REGISTRY_PATH.exists():
        return {"active_document_id": None, "documents": {}}

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry: dict) -> None:
    """Persist indexed document metadata to disk."""
    ensure_storage()

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)


def safe_filename(filename: str) -> str:
    """Normalize uploaded filenames for filesystem storage."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    return cleaned or "uploaded_document.pdf"


def collection_name_for(document_id: str) -> str:
    """Build a ChromaDB collection name for a document."""
    return f"{COLLECTION_PREFIX}_{document_id.replace('-', '_')}"


def infer_document_title(pdf_path: Path, extracted_text: str) -> str:
    """Find a readable title from the first line or filename."""
    file_title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    first_line = extracted_text.splitlines()[0].strip() if extracted_text else ""

    if file_title and first_line.lower().startswith(file_title.lower()):
        return file_title

    if first_line and len(first_line) <= 80:
        return first_line

    return file_title or "Uploaded Document"


def save_upload(file: UploadFile, document_id: str) -> Path:
    """Save the uploaded document to the documents folder."""
    ensure_storage()
    filename = safe_filename(file.filename or "uploaded_document.pdf")
    file_path = DOCUMENTS_DIR / f"{document_id}_{filename}"

    with file_path.open("wb") as output_file:
        output_file.write(file.file.read())

    return file_path


def upload_document(file: UploadFile) -> DocumentResponse:
    """Index one structured or unstructured document and store its metadata."""
    document_id = uuid4().hex
    saved_path = save_upload(file, document_id)
    logger.info("Saved upload %s", saved_path)

    try:
        document_data = read_document(saved_path)
    except DocumentReadError as error:
        raise DocumentProcessingError(str(error)) from error

    extracted_text = document_data["text"]
    if not extracted_text.strip():
        raise DocumentProcessingError(
            "No readable text was found. For scanned PDFs, OCR is required."
        )

    title = infer_document_title(saved_path, extracted_text)
    chunks = split_text_into_chunks(extracted_text)
    model = load_embedding_model()
    chunk_embeddings = create_embeddings(chunks, model)
    collection_name = collection_name_for(document_id)

    create_or_replace_collection(
        collection_name=collection_name,
        chunks=chunks,
        embeddings=chunk_embeddings,
        source_name=file.filename or saved_path.name,
        document_title=title,
    )

    registry = load_registry()
    registry["active_document_id"] = document_id
    registry["documents"][document_id] = {
        "document_id": document_id,
        "filename": file.filename or saved_path.name,
        "title": title,
        "document_kind": document_data["document_kind"],
        "page_count": document_data["page_count"],
        "record_count": document_data["record_count"],
        "chunk_count": len(chunks),
        "status": "ready",
        "file_path": str(saved_path),
        "collection_name": collection_name,
    }
    save_registry(registry)

    logger.info("Indexed document %s with %s chunks", document_id, len(chunks))
    return DocumentResponse(**registry["documents"][document_id])


def list_documents() -> list[DocumentResponse]:
    """List all currently indexed documents."""
    registry = load_registry()
    return [
        DocumentResponse(**metadata)
        for metadata in registry["documents"].values()
    ]


def get_document_metadata(document_id: str | None) -> dict:
    """Resolve requested document metadata or the active document."""
    registry = load_registry()
    resolved_id = document_id or registry.get("active_document_id")

    if not resolved_id or resolved_id not in registry["documents"]:
        raise DocumentNotFoundError("No indexed document was found.")

    return registry["documents"][resolved_id]


def ask_question(
    question: str,
    document_id: str | None = None,
    top_k: int = 3,
) -> AskQuestionResponse:
    """Retrieve context chunks and generate an LLM answer."""
    metadata = get_document_metadata(document_id)
    model = load_embedding_model()
    question_embedding = create_embeddings([question], model)[0]

    retrieved_sections = search_similar_chunks(
        collection_name=metadata["collection_name"],
        query_embedding=question_embedding,
        top_k=top_k,
    )
    retrieved_chunks = [section["text"] for section in retrieved_sections]
    answer = generate_answer(question, retrieved_chunks)

    logger.info("Answered question for document %s", metadata["document_id"])
    return AskQuestionResponse(
        answer=answer,
        document_id=metadata["document_id"],
        question=question,
        retrieved_sections=[
            RetrievedSection(**section) for section in retrieved_sections
        ],
    )


def delete_document(document_id: str) -> None:
    """Delete document metadata, saved file, and ChromaDB collection."""
    registry = load_registry()

    if document_id not in registry["documents"]:
        raise DocumentNotFoundError("The requested document was not found.")

    metadata = registry["documents"].pop(document_id)
    delete_collection(metadata["collection_name"])

    file_path = Path(metadata["file_path"])
    if file_path.exists():
        file_path.unlink()

    if registry.get("active_document_id") == document_id:
        registry["active_document_id"] = next(iter(registry["documents"]), None)

    save_registry(registry)
    logger.info("Deleted document %s", document_id)
