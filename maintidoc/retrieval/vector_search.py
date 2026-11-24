
"""
Recherche vectorielle dans Qdrant.

Ce module expose deux fonctions :
- recherche textuelle dans la collection chunks (BGE-M3, 1024 dimensions) ;
- recherche visuelle dans la collection images (CLIP, 512 dimensions).

Les classements sont fusionnes par RRF dans retriever.py.
"""

from math import isfinite

from maintidoc.vector_store.qdrant_store import (
    search_chunks,
    search_chunks_lexical,
    search_images,
)


def _validate_query_embedding(embedding, expected_dimension: int) -> list[float]:
    """Valide et normalise un vecteur de requête."""
    if embedding is None:
        raise ValueError("L'embedding de requête est obligatoire.")

    try:
        values = [float(value) for value in embedding]
    except (TypeError, ValueError):
        raise ValueError("L'embedding doit contenir uniquement des nombres.") from None

    if len(values) != expected_dimension:
        raise ValueError(
            f"Dimension incorrecte : {len(values)}. "
            f"Dimension attendue : {expected_dimension}."
        )

    if not all(isfinite(value) for value in values):
        raise ValueError("L'embedding contient une valeur non finie.")

    return values


def search_text(
    query_embedding,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Recherche les passages textuels les plus proches du vecteur fourni.

    Args:
        query_embedding: vecteur BGE-M3 de 1024 dimensions.
        top_k: nombre maximal de résultats.
        document_id: identifiant du document (optionnel).

    Returns:
        Liste des passages classés par similarité décroissante.
    """
    embedding = _validate_query_embedding(query_embedding, 1024)

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k doit être un entier strictement positif.")

    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        raise ValueError("document_id doit être une chaîne non vide ou None.")

    return search_chunks(
        query_embedding=embedding,
        top_k=top_k,
        document_id=document_id,
    )


def search_visual(
    query_embedding,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Recherche les images les plus proches du vecteur fourni.

    Args:
        query_embedding: vecteur CLIP de 512 dimensions.
        top_k: nombre maximal de résultats.
        document_id: identifiant du document (optionnel).

    Returns:
        Liste des images classées par similarité décroissante.
    """
    embedding = _validate_query_embedding(query_embedding, 512)

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k doit être un entier strictement positif.")

    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        raise ValueError("document_id doit être une chaîne non vide ou None.")

    return search_images(
        query_embedding=embedding,
        top_k=top_k,
        document_id=document_id,
    )


def search_lexical(query: str, top_k: int = 10, document_id: str | None = None) -> list[dict]:
    """Recherche lexicalement le texte et les tableaux indexés."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("La requête lexicale doit être non vide.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k doit être un entier strictement positif.")
    if document_id is not None and (not isinstance(document_id, str) or not document_id.strip()):
        raise ValueError("document_id doit être une chaîne non vide ou None.")
    return search_chunks_lexical(query, top_k=top_k, document_id=document_id)
