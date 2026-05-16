from pathlib import Path
from typing import Any

import chromadb


VECTOR_STORE_DIR = Path("vector_store")


def get_chroma_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client stored in the project folder."""
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def create_or_replace_collection(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source_name: str,
    document_title: str,
) -> None:
    """Store document chunks and embeddings in a fresh ChromaDB collection."""
    client = get_chroma_client()

    existing_collections = [
        collection.name if hasattr(collection, "name") else collection
        for collection in client.list_collections()
    ]
    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{source_name}-chunk-{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "document_title": document_title,
            "chunk_index": index + 1,
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_collection(collection_name: str) -> None:
    """Delete a ChromaDB collection if it exists."""
    client = get_chroma_client()
    existing_collections = [
        collection.name if hasattr(collection, "name") else collection
        for collection in client.list_collections()
    ]

    if collection_name in existing_collections:
        client.delete_collection(collection_name)


def search_similar_chunks(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search ChromaDB for chunks that are closest to the question embedding."""
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": document,
            "distance": distance,
            "metadata": metadata,
        }
        for document, distance, metadata in zip(documents, distances, metadatas)
    ]
