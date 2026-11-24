from __future__ import annotations

from typing import Any


ALLOWED_TYPES = {
    "text",
    "texte",
    "table",
    "tableau",
}


def validate_chunks(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Vérifie les chunks et distingue les erreurs des chunks invalides."""

    errors = []
    warnings = []
    seen_ids = set()
    invalid_count = 0

    for index, chunk in enumerate(chunks):
        label = f"Chunk à l'index {index}"
        error_count_before = len(errors)

        if not isinstance(chunk, dict):
            errors.append(
                f"{label} : format invalide, dictionnaire attendu."
            )
            invalid_count += 1
            continue

        chunk_id = chunk.get("id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            errors.append(f"{label} : identifiant absent ou invalide.")
        elif chunk_id in seen_ids:
            errors.append(
                f"{label} : identifiant dupliqué : {chunk_id}."
            )
        else:
            seen_ids.add(chunk_id)

        content = chunk.get("text")

        if not isinstance(content, str) or not content.strip():
            errors.append(f"{label} : texte absent ou vide.")

        page = chunk.get("page")

        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
        ):
            errors.append(f"{label} : numéro de page invalide.")

        source_id = chunk.get("source_id")

        if not isinstance(source_id, str) or not source_id.strip():
            warnings.append(f"{label} : source_id absent ou invalide.")

        document = chunk.get("document")

        if not isinstance(document, str) or not document.strip():
            warnings.append(f"{label} : nom du document absent ou invalide.")

        chunk_type = chunk.get("type")

        if (
            not isinstance(chunk_type, str)
            or chunk_type.lower() not in ALLOWED_TYPES
        ):
            errors.append(
                f"{label} : type de contenu non reconnu : {chunk_type}."
            )

        sha256 = chunk.get("sha256")

        if not isinstance(sha256, str) or not sha256.strip():
            warnings.append(f"{label} : empreinte SHA-256 absente.")

        if len(errors) > error_count_before:
            invalid_count += 1

    total = len(chunks)
    valid_count = total - invalid_count

    return {
        "valid": invalid_count == 0,
        "total": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "error_count": len(errors),
        "errors": errors,
        "warnings": warnings,
    }
