
"""Extraction générique des textes, tableaux et éléments visuels PDF.

Fonctionnalités :
- Extraction du texte natif.
- Détection des pages nécessitant potentiellement un OCR.
- OCR français et anglais facultatif.
- Extraction des tableaux.
- Extraction des images intégrées.
- Détection et rendu des pages contenant des dessins vectoriels.
- Conservation des métadonnées et de la provenance.
- Annotations facultatives des images.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"

# Tesseract peut être configuré par variable d'environnement.
# Si aucun chemin n'est défini, PyMuPDF essaiera d'utiliser
# la configuration Tesseract disponible sur la machine.
TESSDATA_DIR = (
    Path(os.environ["TESSDATA_PREFIX"])
    if os.environ.get("TESSDATA_PREFIX")
    else None
)

# Nombre minimal de dessins vectoriels pour déclencher
# le rendu visuel d'une page.
VECTOR_DRAWING_THRESHOLD = 5

# Dimensions minimales des images extraites.
MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 85


def calculate_sha256(data: bytes) -> str:
    """Calcule l'empreinte SHA-256 du PDF."""
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    """Normalise les espaces sans supprimer le contenu."""
    return re.sub(r"\s+", " ", text).strip()


def validate_image_annotations(
    image_annotations: dict[str, dict[str, str | None]] | None,
) -> dict[str, dict[str, str | None]]:
    """Valide les annotations facultatives des images."""

    if image_annotations is None:
        return {}

    if not isinstance(image_annotations, dict):
        raise TypeError(
            "image_annotations doit être un dictionnaire."
        )

    for image_id, annotation in image_annotations.items():

        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(
                "Identifiant d'image invalide dans les annotations."
            )

        if not isinstance(annotation, dict):
            raise TypeError(
                f"L'annotation de {image_id} doit être un dictionnaire."
            )

        unknown_fields = set(annotation) - {
            "figure_id",
            "description",
        }

        if unknown_fields:
            raise ValueError(
                f"Champs d'annotation inconnus pour {image_id} : "
                f"{unknown_fields}"
            )

        for field in ("figure_id", "description"):
            value = annotation.get(field)

            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"{field} doit être une chaîne ou None "
                    f"pour {image_id}."
                )

            if isinstance(value, str) and not value.strip():
                raise ValueError(
                    f"{field} ne doit pas être une chaîne vide "
                    f"pour {image_id}."
                )

    return image_annotations


def _run_ocr(page: Any, native_text: str) -> tuple[str, bool]:
    """
    Tente un OCR sans rendre Tesseract obligatoire.

    Retourne :
    - le texte obtenu ;
    - un booléen indiquant si l'OCR a réussi.

    Une erreur OCR est traitée par l'appelant.
    """

    ocr_arguments = {
        "language": "fra+eng",
        "dpi": 300,
        "full": not bool(native_text),
    }

    # Lire la variable au moment de l'OCR : en ingestion CLI,
    # Le stockage charge parfois le .env apres l'import de ce module.
    configured_tessdata = os.getenv("TESSDATA_PREFIX")
    tessdata_dir = (
        Path(configured_tessdata)
        if configured_tessdata
        else TESSDATA_DIR
    )

    if tessdata_dir is not None:
        if not tessdata_dir.is_dir():
            raise FileNotFoundError(
                "TESSDATA_PREFIX doit désigner le dossier tessdata de Tesseract : "
                f"{tessdata_dir}"
            )
        ocr_arguments["tessdata"] = str(tessdata_dir)

    text_page = page.get_textpage_ocr(**ocr_arguments)

    ocr_text = clean_text(
        page.get_text(
            "text",
            textpage=text_page,
            sort=True,
        )
    )

    return ocr_text, True


def _save_image(
    image: Image.Image,
    image_path: Path,
) -> None:
    """Enregistre une image au format PNG."""
    image_path.parent.mkdir(parents=True, exist_ok=True)

    image.convert("RGB").save(
        image_path,
        format="PNG",
    )


def _extract_embedded_images(
    document: Any,
    page: Any,
    document_id: str,
    document_hash: str,
    filename: str,
    page_number: int,
    output_dir: Path,
    image_annotations: dict[str, dict[str, str | None]],
    errors: list[dict[str, Any]],
    extracted_image_ids: set[str],
) -> list[dict[str, Any]]:
    """Extrait les images intégrées à une page PDF."""

    extracted_images = []
    seen_xrefs = set()

    image_infos = page.get_images(full=True)

    for image_index, image_info in enumerate(
        image_infos,
        start=1,
    ):
        xref = image_info[0]

        if xref in seen_xrefs:
            continue

        seen_xrefs.add(xref)

        try:
            extracted = document.extract_image(xref)
            image_bytes = extracted["image"]

            with Image.open(
                io.BytesIO(image_bytes)
            ) as source_image:
                image = source_image.convert("RGB")

            if (
                image.width < MIN_IMAGE_WIDTH
                or image.height < MIN_IMAGE_HEIGHT
            ):
                continue

            image_id = (
                f"{document_id}-P{page_number:03d}"
                f"-IMG{image_index:02d}"
            )

            image_path = output_dir / f"{image_id}.png"

            _save_image(image, image_path)

            annotation = image_annotations.get(image_id, {})

            image_record = {
                "id": image_id,
                "document_id": document_id,
                "filename": filename,
                "page": page_number,
                "type": "image",
                "path": str(image_path),
                "width": image.width,
                "height": image.height,
                "figure_id": annotation.get("figure_id"),
                "description": annotation.get("description"),
                "sha256": document_hash,
                "metadata": {
                    "resource_type": "original_image",
                    "pdf_xref": xref,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                },
            }

            extracted_images.append(image_record)
            extracted_image_ids.add(image_id)

        except Exception as exc:
            errors.append({
                "page": page_number,
                "component": "image_extraction",
                "xref": xref,
                "error": str(exc),
            })

    return extracted_images


def _render_vector_page(
    page: Any,
    document_id: str,
    document_hash: str,
    filename: str,
    page_number: int,
    output_dir: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Génère une représentation visuelle d'une page
    contenant suffisamment de dessins vectoriels.

    Le rendu représente la page entière, pas une figure isolée.
    """

    try:
        drawings = page.get_drawings()

        if len(drawings) < VECTOR_DRAWING_THRESHOLD:
            return None

        # Résolution de rendu : 150 DPI.
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(150 / 72, 150 / 72),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )

        image_bytes = pixmap.tobytes("png")

        with Image.open(
            io.BytesIO(image_bytes)
        ) as rendered_image:
            image = rendered_image.convert("RGB")

        if image.width <= 0 or image.height <= 0:
            return None

        image_id = (
            f"{document_id}-P{page_number:03d}-PAGE-RENDER"
        )

        image_path = output_dir / f"{image_id}.png"

        _save_image(image, image_path)

        return {
            "id": image_id,
            "document_id": document_id,
            "filename": filename,
            "page": page_number,
            "type": "page_render",
            "path": str(image_path),
            "width": image.width,
            "height": image.height,
            "figure_id": None,
            "description": None,
            "sha256": document_hash,
            "metadata": {
                "resource_type": "page_render",
                "render_method": "vector_page",
                "vector_drawing_count": len(drawings),
                "render_dpi": 150,
            },
        }

    except Exception as exc:
        errors.append({
            "page": page_number,
            "component": "vector_page_render",
            "error": str(exc),
        })

        return None


def _extract_table_cells(page: Any, table: Any) -> list[list[str]]:
    """Reconstruit les cellules à partir des mots et des coordonnées PDF."""
    page_words = page.get_text("words")
    extracted_rows = []

    for table_row in getattr(table, "rows", []):
        extracted_cells = []
        for cell_bbox in table_row.cells:
            if not cell_bbox:
                extracted_cells.append("")
                continue

            x0, y0, x1, y1 = cell_bbox
            words = [
                word for word in page_words
                if x0 - 0.5 <= (word[0] + word[2]) / 2 <= x1 + 0.5
                and y0 - 0.5 <= (word[1] + word[3]) / 2 <= y1 + 0.5
            ]
            words.sort(key=lambda word: (word[1], word[0]))

            lines = []
            current_line = []
            current_y = None
            for word in words:
                y = round((word[1] + word[3]) / 2, 1)
                if current_y is not None and abs(y - current_y) > 2:
                    lines.append(" ".join(current_line))
                    current_line = []
                current_line.append(str(word[4]))
                current_y = y
            if current_line:
                lines.append(" ".join(current_line))

            extracted_cells.append(" ".join(lines).strip())

        extracted_rows.append(extracted_cells)

    if extracted_rows and any(any(cell for cell in row) for row in extracted_rows):
        return extracted_rows

    # Fallback for tables whose cell coordinates are unavailable.
    return [
        [str(value).strip() if value is not None else "" for value in row]
        for row in table.extract()
    ]


def load_pdf(
    pdf_bytes: bytes,
    filename: str,
    image_dir: Path | str | None = None,
    image_annotations: dict[str, dict[str, str | None]] | None = None,
    render_vector_pages: bool = False,
) -> dict[str, Any]:
    """
    Extrait les textes, tableaux et éléments visuels d'un PDF.

    Les images intégrées sont extraites individuellement.

    Les pages contenant suffisamment de dessins vectoriels
    peuvent également être rendues en images complètes.

    L'OCR est facultatif : son échec ne bloque pas
    l'extraction du texte natif.

    Les annotations des images sont facultatives.
    """

    # 1. Validation des entrées.

    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise ValueError(
            "Le fichier PDF est vide ou invalide."
        )

    if (
        not isinstance(filename, str)
        or not filename.strip()
        or not filename.lower().endswith(".pdf")
    ):
        raise ValueError(
            "Le fichier fourni doit être un PDF."
        )

    if not isinstance(render_vector_pages, bool):
        raise TypeError(
            "render_vector_pages doit être un booléen."
        )

    image_annotations = validate_image_annotations(
        image_annotations
    )

    # 2. Préparation des chemins et identifiants.

    output_dir = (
        Path(image_dir) if image_dir else DEFAULT_IMAGE_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_hash = calculate_sha256(pdf_bytes)
    document_id = document_hash[:16]

    texts = []
    tables = []
    images = []
    errors = []

    # 3. Ouverture du PDF.

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

    except Exception as exc:
        raise ValueError(
            f"Impossible d'ouvrir le PDF : {exc}"
        ) from exc

    try:
        if document.is_encrypted:
            raise ValueError(
                "Le PDF est protégé par un mot de passe."
            )

        page_count = len(document)

        if page_count == 0:
            raise ValueError(
                "Le PDF ne contient aucune page."
            )

        extracted_image_ids = set()

        # 4. Traitement de chaque page.

        for page_index, page in enumerate(document):
            page_number = page_index + 1

            # --------------------------------------------
            # 4.1 Extraction du texte natif
            # --------------------------------------------

            native_text = clean_text(
                page.get_text(
                    "text",
                    sort=True,
                )
            )

            image_infos = page.get_images(full=True)

            page_text = native_text
            ocr_attempted = False

            # --------------------------------------------
            # 4.2 Détection et tentative OCR
            # --------------------------------------------

            needs_ocr = (
                len(native_text) < 80
                or bool(image_infos)
            )

            if needs_ocr:
                try:
                    ocr_text, ocr_attempted = _run_ocr(
                        page,
                        native_text,
                    )

                    if len(ocr_text) > len(native_text):
                        page_text = ocr_text

                except Exception as exc:
                    errors.append({
                        "page": page_number,
                        "component": "ocr",
                        "error": str(exc),
                    })

            # --------------------------------------------
            # 4.3 Enregistrement du texte
            # --------------------------------------------

            if page_text:
                texts.append({
                    "id": (
                        f"{document_id}-P{page_number:03d}-TEXT"
                    ),
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "type": "text",
                    "text": page_text,
                    "ocr_attempted": ocr_attempted,
                    "extraction_method": (
                        "native+ocr"
                        if ocr_attempted
                        else "native"
                    ),
                    "sha256": document_hash,
                })

            # --------------------------------------------
            # 4.4 Extraction des tableaux
            # --------------------------------------------

            try:
                table_finder = page.find_tables()

                for table_index, table in enumerate(
                    table_finder.tables,
                    start=1,
                ):
                    rows = _extract_table_cells(page, table)

                    if not rows:
                        continue

                    table_text = "\n".join(
                        " | ".join(row)
                        for row in rows
                    )

                    tables.append({
                        "id": (
                            f"{document_id}-P{page_number:03d}"
                            f"-TABLE{table_index:02d}"
                        ),
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "type": "table",
                        "text": table_text,
                        "rows": rows,
                        "sha256": document_hash,
                    })

            except Exception as exc:
                errors.append({
                    "page": page_number,
                    "component": "table_extraction",
                    "error": str(exc),
                })

            # --------------------------------------------
            # 4.5 Extraction des images intégrées
            # --------------------------------------------

            page_images = _extract_embedded_images(
                document=document,
                page=page,
                document_id=document_id,
                document_hash=document_hash,
                filename=filename,
                page_number=page_number,
                output_dir=output_dir,
                image_annotations=image_annotations,
                errors=errors,
                extracted_image_ids=extracted_image_ids,
            )

            images.extend(page_images)

            # --------------------------------------------
            # 4.6 Rendu des pages vectorielles
            # --------------------------------------------

            if render_vector_pages:
                rendered_page = _render_vector_page(
                    page=page,
                    document_id=document_id,
                    document_hash=document_hash,
                    filename=filename,
                    page_number=page_number,
                    output_dir=output_dir,
                    errors=errors,
                )

                if rendered_page is not None:
                    images.append(rendered_page)

        # 5. Vérification des annotations.

        for image_id in image_annotations:
            if image_id not in extracted_image_ids:
                errors.append({
                    "component": "image_annotation",
                    "image_id": image_id,
                    "error": (
                        "Annotation non associée : "
                        "identifiant absent des images intégrées extraites."
                    ),
                })

    finally:
        document.close()

    # 6. Résultat.

    return {
        "document_id": document_id,
        "filename": filename,
        "sha256": document_hash,
        "page_count": page_count,
        "texts": texts,
        "tables": tables,
        "images": images,
        "errors": errors,
    }
