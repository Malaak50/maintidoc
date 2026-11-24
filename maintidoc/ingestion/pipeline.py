
"""Pipeline d'ingestion des documents PDF pour MaintiDoc."""

from pathlib import Path
from typing import Any

from maintidoc.ingestion.pdf_loader import load_pdf
from maintidoc.ingestion.chunker import chunk_records
from maintidoc.ingestion.chunk_validator import validate_chunks

from maintidoc.embeddings.embedding_generator import generate_embeddings
from maintidoc.embeddings.clip_image_generator import (
    generate_image_embeddings,
)

from maintidoc.vector_store.qdrant_store import (
    get_client,
    save_document,
    save_chunks,
    save_embeddings,
    save_image_embeddings,
    set_document_index_status,
)


def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 80,
    batch_size: int = 2,
    image_annotations: dict[str, dict[str, str | None]] | None = None,
    render_vector_pages: bool = False,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Extrait, prépare et enregistre un PDF dans Qdrant.

    Les contenus textuels et visuels sont traités séparément.

    Les annotations des images sont facultatives et indépendantes
    du domaine du document.

    Les donnees sont enregistrees dans les collections Qdrant.

    """

    # 1. Validation des paramètres.

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size doit être un entier.")

    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("overlap doit être un entier.")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size doit être un entier.")

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être positif.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap doit être positif ou nul et inférieur à chunk_size."
        )

    if batch_size <= 0:
        raise ValueError("batch_size doit être positif.")

    if not isinstance(render_vector_pages, bool):
        raise TypeError("render_vector_pages doit être un booléen.")

    # 2. Vérification du fichier PDF.

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Fichier PDF introuvable : {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            "Le chemin fourni ne désigne pas un fichier."
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Le fichier fourni doit être un PDF."
        )

    source_filename = pdf_path.name
    if filename is not None:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename doit être un nom de fichier non vide.")
        source_filename = Path(filename).name
        if not source_filename.lower().endswith(".pdf"):
            raise ValueError("filename doit se terminer par .pdf.")

    # 3. Lecture et extraction du PDF.

    pdf_bytes = pdf_path.read_bytes()

    extracted = load_pdf(
        pdf_bytes=pdf_bytes,
        filename=source_filename,
        image_annotations=image_annotations,
        render_vector_pages=render_vector_pages,
    )

    required_fields = {
        "document_id",
        "filename",
        "sha256",
        "page_count",
        "texts",
        "tables",
        "images",
        "errors",
    }

    missing_fields = required_fields - extracted.keys()

    if missing_fields:
        raise ValueError(
            "Résultat d'extraction incomplet : "
            + ", ".join(sorted(missing_fields))
        )

    document_id = extracted["document_id"]
    texts = extracted["texts"]
    tables = extracted["tables"]
    images = extracted["images"]
    original_images = [
        image for image in images
        if image.get("type", "image") == "image"
    ]
    page_renders = [
        image for image in images
        if image.get("type") == "page_render"
    ]
    ocr_page_count = sum(
        1 for text in texts if text.get("ocr_attempted")
    )
    ocr_error_count = sum(
        1 for error in extracted["errors"]
        if error.get("component") == "ocr"
    )

    # 4. Préparation des contenus textuels.

    records = [*texts, *tables]

    chunks = chunk_records(
        records=records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # 5. Validation des segments.

    validation = validate_chunks(chunks)

    if not validation["valid"]:
        raise ValueError(
            "Validation des segments échouée : "
            + " | ".join(validation["errors"])
        )

    valid_chunks = chunks

    # 6. Génération des embeddings textuels.

    text_embeddings = None

    if valid_chunks:
        chunk_texts = [
            chunk["text"] for chunk in valid_chunks
        ]

        text_embeddings = generate_embeddings(
            chunk_texts,
            batch_size=batch_size,
        )

        if len(text_embeddings) != len(valid_chunks):
            raise ValueError(
                "Le nombre d'embeddings textuels ne correspond "
                "pas au nombre de chunks."
            )

    # 7. Génération des embeddings visuels.

    image_embeddings = None

    if images:
        image_paths = [
            image["path"] for image in images
        ]

        image_embeddings = generate_image_embeddings(
            image_paths,
            batch_size=batch_size,
        )

        if len(image_embeddings) != len(images):
            raise ValueError(
                "Le nombre d'embeddings visuels ne correspond "
                "pas au nombre d'images."
            )

    # 8. Initialisation des compteurs.

    saved_chunks = 0
    saved_text_embeddings = 0
    saved_images = 0

    # 9. Ecriture dans les collections Qdrant.
    get_client()
    save_document(
        document_id=document_id,
        filename=extracted["filename"],
        sha256=extracted["sha256"],
        revision=None,
        page_count=extracted["page_count"],
        metadata={
            "source": "pdf_ingestion",
            "text_count": len(texts),
            "table_count": len(tables),
            "image_count": len(original_images),
            "page_render_count": len(page_renders),
            "ocr_page_count": ocr_page_count,
            "ocr_error_count": ocr_error_count,
        },
    )

    try:
        if valid_chunks:
            saved_chunks = save_chunks(document_id, valid_chunks)
            saved_text_embeddings = save_embeddings(
                document_id, [chunk["id"] for chunk in valid_chunks], text_embeddings,
            )

        if images:
            saved_images = save_image_embeddings(document_id, images, image_embeddings)

        set_document_index_status(document_id, "indexed")
    except Exception as exc:
        set_document_index_status(document_id, "failed", f"{type(exc).__name__}: {exc}")
        raise

    # 10. Résultat de l'ingestion.

    return {
        "document_id": document_id,
        "filename": extracted["filename"],
        "page_count": extracted["page_count"],
        "text_count": len(texts),
        "table_count": len(tables),
        "chunk_count": len(valid_chunks),
        "image_count": len(original_images),
        "total_image_resources": len(images),
        "original_image_count": len(original_images),
        "page_render_count": len(page_renders),
        "ocr_page_count": ocr_page_count,
        "ocr_error_count": ocr_error_count,
        "saved_chunks": saved_chunks,
        "saved_text_embeddings": saved_text_embeddings,
        "saved_images": saved_images,
        "saved_original_images": len(original_images),
        "saved_page_renders": len(page_renders),
        "validation": validation,
        "extraction_errors": extracted["errors"],
    }
