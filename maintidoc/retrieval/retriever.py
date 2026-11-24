
"""
MaintiDoc — orchestration de la récupération documentaire.

Coordonne :
- la recherche vectorielle textuelle (BGE-M3) ;
- la recherche vectorielle visuelle (CLIP) ;
- la préparation des preuves documentaires.

Fusionne la recherche vectorielle, lexicale et visuelle par RRF.
"""

from pathlib import Path
from typing import Any
import re
import unicodedata

from maintidoc.retrieval.vector_search import search_lexical, search_text, search_visual


MIN_IMAGE_SIMILARITY = 0.26
MIN_PAGE_MATCH_IMAGE_SIMILARITY = 0.22


def _terms(value: str) -> set[str]:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).casefold()
    stopwords = {
        "que", "quel", "quelle", "quels", "quelles", "est", "sont", "dans", "pour",
        "avec", "sur", "les", "des", "une", "un", "du", "de", "la", "le", "et",
        "comment", "donne", "donner", "moi", "this", "that", "what", "which", "show",
        "list", "the", "for", "with", "from", "are", "was", "were", "how",
    }
    return {word for word in re.findall(r"[a-z0-9]+", value) if len(word) > 2 and word not in stopwords}


def rerank_text_results(results: list[dict], query: str) -> list[dict]:
    """Reclasse les candidats vectoriels avec couverture lexicale et codes exacts."""
    query_terms = _terms(query)
    code_pattern = re.compile(r"\b(?:AL[- ]?\d{2}|FIG[- ]?\d{2}|INC[- ]?\d{3}|[A-Z]{2,}(?:-[A-Z0-9]+)+)\b", re.I)
    query_codes = {re.sub(r"[ -]", "", code).casefold() for code in code_pattern.findall(query)}
    ranked = []
    max_rrf = max((float(item.get("rrf_score") or 0) for item in results), default=0.0)
    for original in results:
        item = dict(original)
        content = str(item.get("content") or "")
        terms = _terms(content)
        coverage = len(query_terms & terms) / max(1, len(query_terms))
        content_codes = {re.sub(r"[ -]", "", code).casefold() for code in code_pattern.findall(content)}
        code_match = bool(query_codes & content_codes)
        try:
            similarity = float(item.get("similarity") or 0)
        except (TypeError, ValueError):
            similarity = 0.0
        item["lexical_coverage"] = round(coverage, 4)
        item["reference_match"] = code_match
        rrf_score = float(item.get("rrf_score") or 0)
        normalized_rrf = rrf_score / max_rrf if max_rrf else 0.0
        item["hybrid_score"] = round(
            0.60 * normalized_rrf + 0.20 * similarity + 0.15 * coverage + (0.35 if code_match else 0),
            6,
        )
        ranked.append(item)
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    weights: list[float] | None = None,
    rank_constant: int = 60,
) -> list[dict]:
    """Fusionne plusieurs classements par Reciprocal Rank Fusion (RRF)."""
    if rank_constant <= 0:
        raise ValueError("rank_constant doit être positif.")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(weight < 0 for weight in weights):
        raise ValueError("Il faut un poids positif ou nul pour chaque classement.")

    fused: dict[str, dict] = {}
    for channel_index, ranking in enumerate(rankings):
        channel = ("text_vector", "text_lexical", "visual")[channel_index] if channel_index < 3 else f"channel_{channel_index}"
        for rank, result in enumerate(ranking, start=1):
            identity = result.get("chunk_id") or result.get("image_id") or result.get("id")
            if identity is None:
                continue
            key = str(identity)
            if key not in fused:
                fused[key] = dict(result)
                fused[key]["rrf_score"] = 0.0
                fused[key]["retrieval_channels"] = []
            else:
                # Conserve tous les champs disponibles sans écraser une meilleure similarité.
                for field, value in result.items():
                    if value is not None and (fused[key].get(field) is None or field in {"metadata", "description", "figure_id"}):
                        fused[key][field] = value
            fused[key]["rrf_score"] += float(weights[channel_index]) / (rank_constant + rank)
            fused[key]["retrieval_channels"].append(channel)
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def filter_relevant_images(
    image_results: list[dict],
    min_similarity: float = MIN_IMAGE_SIMILARITY,
    supporting_pages: set[int] | None = None,
    query: str = "",
    supporting_text: str = "",
) -> list[dict]:
    """Retient les images CLIP fortes ou concordant avec une page textuelle.

    Si aucune image n'atteint le seuil CLIP standard, une seule image peut
    être retenue à seuil réduit quand sa page apparaît dans les preuves texte.
    """
    relevant = []
    query_terms = _terms(query)
    reference_ids = {value.casefold() for value in re.findall(r"\bFIG[- ]?\d{2}\b", query, re.I)}
    for result in image_results:
        try:
            score = float(result.get("similarity"))
        except (TypeError, ValueError):
            continue
        page = int(result.get("page") or 0)
        page_match = bool(supporting_pages and page in supporting_pages)
        figure_id = str(result.get("figure_id") or "")
        metadata = result.get("metadata") or {}
        caption = " ".join((str(result.get("description") or ""), str(metadata.get("figure_caption") or "")))
        caption_terms = _terms(caption)
        caption_match = len(query_terms & caption_terms) / max(1, len(query_terms))
        ref_match = bool(figure_id and any(re.sub(r"[ -]", "", ref).casefold() == re.sub(r"[ -]", "", figure_id).casefold() for ref in reference_ids))
        item = dict(result)
        item["page_match"] = page_match
        item["caption_match"] = round(caption_match, 4)
        item["visual_relevance"] = round(score + (0.12 if page_match else 0) + 0.2 * caption_match + (0.3 if ref_match else 0), 6)
        if (page_match and score >= MIN_PAGE_MATCH_IMAGE_SIMILARITY) or (score >= min_similarity and not supporting_pages):
            relevant.append(item)

    if relevant:
        return sorted(relevant, key=lambda result: result["visual_relevance"], reverse=True)[:3]

    if not supporting_pages:
        return []

    page_matched = []
    for result in image_results:
        try:
            score = float(result.get("similarity"))
            page = int(result.get("page"))
        except (TypeError, ValueError):
            continue

        if score >= MIN_PAGE_MATCH_IMAGE_SIMILARITY and page in supporting_pages:
            page_matched.append(result)

    if not page_matched:
        return []

    return sorted(page_matched, key=lambda result: float(result["similarity"]), reverse=True)[:3]


def retrieve_evidence(
    text_embedding: Any,
    image_embedding: Any = None,
    top_k_text: int = 5,
    top_k_images: int = 5,
    document_id: str | None = None,
    query: str | None = None,
) -> dict:
    """
    Récupère les passages textuels et les images pertinents.

    Args:
        text_embedding: vecteur BGE-M3 de 1024 dimensions.
        image_embedding: vecteur CLIP de 512 dimensions, optionnel.
        top_k_text: nombre maximal de passages textuels.
        top_k_images: nombre maximal d'images.
        document_id: identifiant du document à filtrer (optionnel).

    Returns:
        Dictionnaire contenant les résultats textuels et visuels.
    """
    vector_text_results = search_text(
        query_embedding=text_embedding,
        top_k=max(top_k_text * 3, top_k_text),
        document_id=document_id,
    )
    lexical_text_results = search_lexical(
        query=query,
        top_k=max(top_k_text * 3, top_k_text),
        document_id=document_id,
    ) if query and query.strip() else []

    image_results = []

    if image_embedding is not None:
        image_results = search_visual(
            query_embedding=image_embedding,
            top_k=top_k_images,
            document_id=document_id,
        )

    fused_results = reciprocal_rank_fusion(
        [vector_text_results, lexical_text_results, image_results],
        weights=[1.0, 1.25, 0.75],
    )
    text_results = [
        item for item in fused_results
        if item.get("chunk_id") is not None
    ][:top_k_text]
    ranked_images = [
        item for item in fused_results
        if item.get("image_id") is not None
    ][:top_k_images]

    return {
        "text_results": text_results,
        "image_results": ranked_images,
        "fused_results": fused_results,
        "text_count": len(text_results),
        "image_count": len(image_results),
        "document_id": document_id,
    }


def prepare_evidence(retrieval_results: dict) -> list[dict]:
    """
    Uniformise les résultats textuels et visuels en preuves documentaires.

    Les références et métadonnées originales sont conservées.
    """
    if not isinstance(retrieval_results, dict):
        raise TypeError("retrieval_results doit être un dictionnaire.")

    evidence = []

    for result in retrieval_results.get("text_results", []):
        chunk_type = result.get("chunk_type") or "text"
        evidence.append({
            "id": result.get("chunk_id"),
            "document_id": result.get("document_id"),
            "page": result.get("page"),
            "type": chunk_type,
            "chunk_type": chunk_type,
            "content": result.get("content"),
            "similarity": result.get("similarity"),
            "rrf_score": result.get("rrf_score"),
            "retrieval_channels": result.get("retrieval_channels", []),
            "lexical_score": result.get("lexical_score"),
            "metadata": result.get("metadata", {}),
        })

    for result in retrieval_results.get("image_results", []):
        evidence.append({
            "id": result.get("image_id"),
            "image_id": result.get("image_id"),
            "document_id": result.get("document_id"),
            "page": result.get("page"),
            "type": "image",
            "image_path": result.get("image_path"),
            "filename": result.get("filename"),
            "width": result.get("width"),
            "height": result.get("height"),
            "figure_id": result.get("figure_id"),
            "description": result.get("description"),
            "similarity": result.get("similarity"),
            "rrf_score": result.get("rrf_score"),
            "retrieval_channels": result.get("retrieval_channels", []),
            "metadata": result.get("metadata", {}),
        })

    return evidence
