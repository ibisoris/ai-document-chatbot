from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services import document_service


client = TestClient(app)


def test_upload_document_endpoint(monkeypatch, tmp_path):
    """Verify that supported uploads are indexed and returned as document metadata."""
    monkeypatch.setattr(document_service, "DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(
        document_service,
        "REGISTRY_PATH",
        tmp_path / "vector_store" / "documents.json",
    )
    monkeypatch.setattr(document_service, "load_embedding_model", lambda: object())
    monkeypatch.setattr(
        document_service,
        "create_embeddings",
        lambda texts, model: [[0.1, 0.2, 0.3] for _ in texts],
    )
    monkeypatch.setattr(
        document_service,
        "create_or_replace_collection",
        lambda **kwargs: None,
    )

    sample_file = Path("sample_documents/sample_policy.txt")

    with sample_file.open("rb") as file:
        response = client.post(
            "/upload-document",
            files={"file": ("sample_policy.txt", file, "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample_policy.txt"
    assert payload["document_kind"] == "unstructured"
    assert payload["chunk_count"] >= 1
    assert payload["status"] == "ready"


def test_ask_question_endpoint(monkeypatch, tmp_path):
    """Verify that questions return an answer and retrieved source sections."""
    registry_path = tmp_path / "vector_store" / "documents.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
        {
          "active_document_id": "doc-1",
          "documents": {
            "doc-1": {
              "document_id": "doc-1",
              "filename": "sample_policy.txt",
              "title": "Remote Work Policy",
              "document_kind": "unstructured",
              "page_count": null,
              "record_count": null,
              "chunk_count": 1,
              "status": "ready",
              "file_path": "documents/sample_policy.txt",
              "collection_name": "document_doc_1"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(document_service, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(document_service, "load_embedding_model", lambda: object())
    monkeypatch.setattr(document_service, "create_embeddings", lambda texts, model: [[0.1]])
    monkeypatch.setattr(
        document_service,
        "search_similar_chunks",
        lambda collection_name, query_embedding, top_k: [
            {
                "text": "Employees may work remotely up to three days per week.",
                "distance": 0.12,
                "metadata": {
                    "source": "sample_policy.txt",
                    "document_title": "Remote Work Policy",
                    "chunk_index": 1,
                },
            }
        ],
    )
    monkeypatch.setattr(
        document_service,
        "generate_answer",
        lambda question, chunks: "Employees may work remotely up to three days per week.",
    )

    response = client.post(
        "/ask-question",
        json={"question": "How many remote days are allowed?", "document_id": "doc-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "doc-1"
    assert "three days" in payload["answer"]
    assert len(payload["retrieved_sections"]) == 1
