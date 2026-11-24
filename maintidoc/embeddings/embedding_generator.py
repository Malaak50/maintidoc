from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 2

_model: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Charge BGE-M3 une seule fois par processus."""
    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            local_files_only=True,
        )
        # Les chunks de MaintiDoc sont courts; limiter la sequence reduit la RAM.
        _model.max_seq_length = 256

    return _model


def generate_embeddings(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Génère des embeddings textuels normalisés de dimension 1024."""

    if not isinstance(texts, list):
        raise TypeError("texts doit être une liste.")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size doit être un entier strictement positif.")

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Tous les textes doivent être des chaînes non vides."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        [text.strip() for text in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_shape = (len(texts), EMBEDDING_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Dimensions inattendues : {embeddings.shape}. "
            f"Dimensions attendues : {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs non finies."
        )

    return embeddings
