from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a sentence-transformers model for creating text embeddings."""
    return SentenceTransformer(model_name)


def create_embeddings(
    texts: list[str],
    model: SentenceTransformer,
) -> list[list[float]]:
    """Convert text chunks into numeric vectors that ChromaDB can search."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
