"""
embeddings.py
Wraps a Sentence-Transformers model for embedding text (and image captions,
which are just text once the vision model has described them).
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, solid for RAG


def format_embedding_error(exc: Exception) -> str:
    message = str(exc)
    return (
        "Failed to initialize the embedding model. "
        f"Original error: {message}\n"
        "This usually means the model was not cached locally or the network is blocking Hugging Face. "
        "If you are on a restricted network, connect once to an unrestricted network and run: "
        f"'python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer(\'{EMBEDDING_MODEL_NAME}\')\"' "
        "or set HF_HUB_OFFLINE=1 after the model is cached."
    )


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(format_embedding_error(exc)) from exc
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        try:
            vectors = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        except Exception as exc:
            raise RuntimeError(format_embedding_error(exc)) from exc
        return vectors.astype("float32")
