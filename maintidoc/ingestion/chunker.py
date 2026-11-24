
"""Découpage des contenus PDF pour l'indexation RAG."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    """Nettoie les espaces sans supprimer la structure des lignes."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """Découpe un texte en segments avec chevauchement contrôlé."""

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à zéro.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap doit être positif ou nul et inférieur à chunk_size."
        )

    text = normalize_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            newline = text.rfind("\n", start, end)
            boundary = max(boundary, newline)

            if boundary > start + chunk_size // 2:
                end = boundary

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def split_table(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """Découpe un tableau par groupes de lignes complètes."""

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à zéro.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap doit être positif ou nul et inférieur à chunk_size."
        )

    lines = [
        line.strip()
        for line in normalize_text(text).splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:
        if len(line) > chunk_size:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_length = 0

            chunks.extend(split_text(line, chunk_size, overlap))
            continue

        additional_length = len(line) + (1 if current_lines else 0)

        if current_lines and current_length + additional_length > chunk_size:
            chunks.append("\n".join(current_lines))

            # Réutilise les dernières lignes comme contexte.
            retained = []
            retained_length = 0

            for previous_line in reversed(current_lines):
                cost = len(previous_line) + 1

                if retained_length + cost > overlap:
                    break

                retained.insert(0, previous_line)
                retained_length += cost

            current_lines = retained
            current_length = len("\n".join(current_lines))

        current_lines.append(line)
        current_length = len("\n".join(current_lines))

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """Découpe les enregistrements en conservant leur provenance."""

    results = []

    for record in records:
        content = record.get("text", "")

        if not isinstance(content, str) or not content.strip():
            continue

        record_type = str(record.get("type", "texte")).lower()
        base_id = record.get("id", "UNKNOWN")

        if record_type in {"table", "tableau"}:
            pieces = split_table(content, chunk_size, overlap)
        else:
            pieces = split_text(content, chunk_size, overlap)

        for index, piece in enumerate(pieces, start=1):
            chunk = {
                "id": f"{base_id}-CH{index:03d}",
                "source_id": base_id,
                "page": record.get("page"),
                "type": record_type,
                "text": piece,
                "document": record.get("document") or record.get("filename"),
                "sha256": record.get("sha256", record.get("sha")),
                "chunk_index": index,
            }

            if "ocr_attempted" in record:
                chunk["ocr_attempted"] = bool(record["ocr_attempted"])
            if record.get("extraction_method"):
                chunk["extraction_method"] = record["extraction_method"]

            if record_type in {"table", "tableau"}:
                chunk["table_rows"] = [
                    [cell.strip() for cell in line.split("|")]
                    for line in piece.splitlines()
                    if line.strip()
                ]

            results.append(chunk)

    return results
