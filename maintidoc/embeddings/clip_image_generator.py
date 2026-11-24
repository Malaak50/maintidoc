"""Génération des embeddings visuels MaintiDoc avec CLIP."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
EMBEDDING_DIMENSION = 512
DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 2

_model: SentenceTransformer | None = None


def load_clip_model() -> SentenceTransformer:
    """Charge CLIP une seule fois depuis le cache local."""

    global _model

    if _model is None:
        model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            local_files_only=True,
        )

        dimension = model.get_embedding_dimension()

        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimension CLIP incorrecte : {dimension}. "
                f"Dimension attendue : {EMBEDDING_DIMENSION}."
            )

        _model = model

    return _model


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Ouvre une image locale et la convertit en RGB."""

    if not isinstance(image_path, (str, Path)):
        raise TypeError("Le chemin de l'image doit être une chaîne ou un Path.")

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {path}")

    if not path.is_file():
        raise ValueError(f"Le chemin indiqué n'est pas un fichier : {path}")

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Impossible de lire l'image {path} : {exc}"
        ) from exc


def generate_image_embeddings(
    image_paths: list[Union[str, Path]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Génère des embeddings CLIP normalisés de dimension 512."""

    if not isinstance(image_paths, list):
        raise TypeError("image_paths doit être une liste.")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size doit être un entier strictement positif.")

    if not image_paths:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    # Lire et valider les images avant de charger le modèle.
    images = [
        load_image(image_path)
        for image_path in image_paths
    ]

    model = load_clip_model()

    embeddings = model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_shape = (len(image_paths), EMBEDDING_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Dimensions inattendues : {embeddings.shape}. "
            f"Dimensions attendues : {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Les embeddings contiennent des valeurs non finies.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Les embeddings ne sont pas correctement normalisés.")

    return embeddings


def generate_single_image_embedding(
    image_path: Union[str, Path],
) -> np.ndarray:
    """Génère l'embedding CLIP d'une seule image."""

    embeddings = generate_image_embeddings(
        [image_path],
        batch_size=1,
    )

    return embeddings[0]
