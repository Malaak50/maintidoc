
"""MaintiDoc — interface Streamlit Qdrant et Qwen local."""
from __future__ import annotations

import importlib
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
from urllib.request import urlopen
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Permet de lancer l'application depuis n'importe quel répertoire.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "maintidoc-qwen")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")

TOP_K_TEXT = 4
TOP_K_IMAGES = 6

MAX_IMAGES_TO_VLM = 1
MAX_IMAGES_TO_DISPLAY = 4

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_EVIDENCE_CHARS = 4500

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

SUPPORTED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

# ============================================================
# 2. IMPORTS DU PROJET
# ============================================================

from maintidoc.ingestion.pipeline import ingest_pdf
from maintidoc.embeddings.embedding_generator import generate_embeddings
from maintidoc.retrieval.retriever import (
    retrieve_evidence,
    prepare_evidence,
    filter_relevant_images,
    rerank_text_results,
)
from maintidoc.agent.workflow import build_workflow, classify_question
from maintidoc.mcp.client import call_maintidoc_tool
from maintidoc.generation.qwen import ask_qwen


# ============================================================
# 3. CONNEXION ET INVENTAIRE DES DOCUMENTS
# ============================================================

def get_vector_documents():
    """Renvoie l'inventaire des manuels de la collection Qdrant."""
    from maintidoc.vector_store.qdrant_store import list_documents
    return list_documents()


@st.cache_data(ttl=10, max_entries=1)
def get_local_llm_status() -> tuple[bool, str]:
    """Sonde l'endpoint de découverte de llama.cpp, pas la racine /v1."""
    try:
        with urlopen(f"{LOCAL_LLM_BASE_URL}/models", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data", [])
        names = ", ".join(str(model.get("id", "")) for model in models if model.get("id"))
        return True, names or "API joignable"
    except Exception as exc:
        return False, type(exc).__name__


# ============================================================
# 4. OUTILS
# ============================================================

def to_plain(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def safe_page(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_prepared_evidence(prepared, query: str = "") -> dict:
    """Normalise la sortie de prepare_evidence()."""

    if isinstance(prepared, list):
        text_results = []
        image_results = []

        for item in prepared:
            if not isinstance(item, dict):
                continue

            if item.get("image_path") or item.get("image_id"):
                image_results.append(item)
            else:
                text_results.append(item)

        prepared = {
            "text_results": text_results,
            "image_results": image_results,
        }

    if not isinstance(prepared, dict):
        raise TypeError(
            "prepare_evidence() doit renvoyer un dictionnaire "
            "ou une liste."
        )

    prepared.setdefault("text_results", [])
    prepared.setdefault("image_results", [])
    prepared["text_results"] = rerank_text_results(prepared["text_results"], query)

    captions_by_page = {}
    for text_item in prepared["text_results"]:
        page = safe_page(text_item.get("page"))
        content = str(text_item.get("content") or "")
        for match in re.finditer(r"\bFIG[- ]?\d{2}\s*[:–-]\s*([^\n.;]{3,180})", content, re.I):
            if page:
                captions_by_page.setdefault(page, []).append((match.group(0).strip(), match.group(1).strip()))
    for image in prepared["image_results"]:
        page = safe_page(image.get("page"))
        captions = captions_by_page.get(page, [])
        if captions:
            metadata = dict(image.get("metadata") or {})
            caption, description = captions[0]
            if not image.get("figure_id"):
                image["figure_id"] = caption.split(":", 1)[0].strip().replace(" ", "")
            if not image.get("description"):
                image["description"] = description
            metadata.setdefault("figure_caption", caption)
            image["metadata"] = metadata

    unique_images = {}

    for item in prepared["image_results"]:
        if not isinstance(item, dict):
            continue

        image_id = item.get("image_id")

        if image_id is None:
            continue

        key = str(image_id)

        try:
            score = float(item.get("similarity") or 0)
        except (TypeError, ValueError):
            score = 0

        previous = unique_images.get(key)

        if previous is None:
            unique_images[key] = item
        else:
            try:
                previous_score = float(
                    previous.get("similarity") or 0
                )
            except (TypeError, ValueError):
                previous_score = 0

            if score > previous_score:
                unique_images[key] = item

    prepared["image_results"] = sorted(
        unique_images.values(),
        key=lambda item: float(
            item.get("similarity") or 0
        ),
        reverse=True,
    )
    prepared["image_results"] = filter_relevant_images(
        prepared["image_results"],
        supporting_pages={safe_page(item.get("page")) for item in prepared["text_results"] if safe_page(item.get("page")) > 0},
        query=query,
    )

    return prepared


def render_bordered_table(table_rows, fallback_content: str) -> None:
    """Affiche un tableau HTML avec bordures, y compris les anciens chunks aplatis."""
    rows = table_rows or []

    if not rows:
        rows = []
        pending_cell = None
        for line in str(fallback_content or "").splitlines():
            if "|" not in line:
                if line.strip():
                    pending_cell = line.strip()
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if pending_cell and cells and cells[0] in {"", "-", "–", "—"}:
                cells[0] = pending_cell
            pending_cell = None
            rows.append(cells)

    if not rows:
        st.markdown(str(fallback_content or ""))
        return

    merged_rows = []
    pending_cell = None
    for row in rows:
        cells = [str(cell).strip() for cell in row]
        if len(cells) == 1 and merged_rows:
            pending_cell = cells[0]
            continue
        if pending_cell and cells and cells[0] in {"", "-", "–", "—"}:
            cells[0] = pending_cell
        pending_cell = None
        merged_rows.append(cells)

    width = max(len(row) for row in merged_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in merged_rows]

    if len(normalized_rows) > 1:
        frame = pd.DataFrame(
            normalized_rows[1:],
            columns=normalized_rows[0],
        )
    else:
        frame = pd.DataFrame(normalized_rows)

    table_html = frame.to_html(
        index=False,
        border=0,
        escape=True,
        classes="maintidoc-bordered-table",
    )
    st.markdown(
        "<style>"
        ".maintidoc-bordered-table{border-collapse:collapse;width:100%;}"
        ".maintidoc-bordered-table th,.maintidoc-bordered-table td{"
        "border:1px solid #9aa4b2;padding:7px 10px;text-align:left;}"
        ".maintidoc-bordered-table th{background:#eef2f7;}"
        "</style>"
        f"<div style='overflow-x:auto'>{table_html}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 5. STATISTIQUES D'INGESTION
# ============================================================

def show_ingestion_metrics(result: dict):
    """Affiche les statistiques retournées par ingest_pdf()."""

    st.markdown("#### Statistiques du manuel indexé")

    metrics = [
        ("Pages", result.get("page_count", 0)),
        ("Textes extraits", result.get("text_count", 0)),
        ("Tableaux extraits", result.get("table_count", 0)),
        ("Segments préparés", result.get("chunk_count", 0)),
        ("Segments enregistrés", result.get("saved_chunks", 0)),
        (
            "Embeddings textuels enregistrés",
            result.get("saved_text_embeddings", 0),
        ),
        ("Images originales extraites", result.get("original_image_count", result.get("image_count", 0))),
        ("Rendus de pages extraits", result.get("page_render_count", 0)),
        ("Images originales enregistrées", result.get("saved_original_images", result.get("saved_images", 0))),
        ("Rendus enregistrés", result.get("saved_page_renders", 0)),
    ]

    columns = st.columns(4)

    for index, (label, value) in enumerate(metrics):
        columns[index % 4].metric(
            label,
            value if value is not None else "—",
        )

    errors = result.get("extraction_errors") or []

    if errors:
        with st.expander("Erreurs d'extraction"):
            for error in errors:
                st.warning(str(error))

    validation = result.get("validation")

    if validation:
        with st.expander("Résultats de validation"):
            st.json(validation)


# ============================================================
# 6. EMBEDDINGS DE LA QUESTION
# ============================================================

@st.cache_resource(show_spinner="Chargement du modèle CLIP...")
def load_clip_model():
    """Charge CLIP localement sur CPU."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        CLIP_MODEL_NAME,
        device="cpu",
        local_files_only=True,
    )


def make_query_embeddings(question: str):
    # BGE-M3 : 1024 dimensions.

    text_vector = np.asarray(
        generate_embeddings([question])[0],
        dtype=np.float32,
    )

    # CLIP : 512 dimensions.

    clip = load_clip_model()

    image_vector = np.asarray(
        clip.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0],
        dtype=np.float32,
    )

    if text_vector.shape != (1024,):
        raise ValueError(
            f"Embedding BGE-M3 inattendu : {text_vector.shape}"
        )

    if image_vector.shape != (512,):
        raise ValueError(
            f"Embedding CLIP inattendu : {image_vector.shape}"
        )

    if (
        not np.isfinite(text_vector).all()
        or not np.isfinite(image_vector).all()
    ):
        raise ValueError(
            "Un embedding contient une valeur non finie."
        )

    return text_vector, image_vector


# ============================================================
# 7. RECHERCHE QDRANT
# ============================================================

def retrieve(
    question: str,
    document_id: str | None = None,
) -> dict:

    text_vector, image_vector = make_query_embeddings(
        question
    )

    raw = retrieve_evidence(
        text_embedding=text_vector,
        image_embedding=image_vector,
        top_k_text=TOP_K_TEXT,
        top_k_images=TOP_K_IMAGES,
        document_id=document_id,
        query=question,
    )

    return normalize_prepared_evidence(
        prepare_evidence(raw), query=question
    )


# ============================================================
# 8. PRÉPARATION DES PREUVES ET DES IMAGES
# ============================================================

def build_vlm_payload(prepared: dict):
    """
    Prépare les extraits textuels et les images accessibles.
    """

    evidence = []
    image_payloads = []

    used_chars = 0

    # -------------------------
    # Résultats textuels
    # -------------------------

    for item in prepared.get("text_results", []):

        content = str(
            item.get("content") or ""
        ).strip()

        if not content:
            continue

        remaining = MAX_EVIDENCE_CHARS - used_chars

        if remaining <= 0:
            break

        content = content[:remaining]
        used_chars += len(content)

        chunk_id = (
            item.get("chunk_id")
            or item.get("id")
        )

        if chunk_id is None:
            continue

        evidence.append({
            "id": str(chunk_id),
            "page": safe_page(item.get("page")),
            "type": str(
                item.get("type")
                or item.get("chunk_type")
                or "texte"
            ),
            "content": content,
        })

    # -------------------------
    # Résultats images
    # -------------------------

    for item in prepared.get("image_results", []):

        image_id = item.get("image_id")
        image_path = item.get("image_path")

        if image_id is None or not image_path:
            continue

        path = Path(str(image_path))

        if not path.is_file():
            continue

        mime_type, _ = mimetypes.guess_type(path.name)

        if mime_type not in SUPPORTED_IMAGE_MIMES:
            continue

        try:
            data = path.read_bytes()
        except OSError:
            continue

        if not data or len(data) > MAX_IMAGE_BYTES:
            continue

        metadata = item.get("metadata") or {}
        image_kind = (
            metadata.get("resource_type") or metadata.get("type")
            if isinstance(metadata, dict)
            else None
        )
        if image_kind not in {"original_image", "page_render"}:
            image_kind = (
                "page_render"
                if str(image_id).endswith("-PAGE-RENDER")
                else "original_image"
            )
        source_document = (
            metadata.get("source_pdf_filename")
            if isinstance(metadata, dict)
            else None
        ) or item.get("document_id")

        image_payloads.append({
            "id": str(image_id),
            "page": safe_page(item.get("page")),
            "resource_type": image_kind,
            "source_document": source_document,
            "bytes": data,
            "mime_type": mime_type,
            "image_path": str(path),
            "description": str(
                item.get("description")
                or (
                    "Rendu de page du manuel."
                    if image_kind == "page_render"
                    else "Image originale extraite du manuel."
                )
            ),
            "similarity": to_plain(
                item.get("similarity")
            ),
            "figure_id": item.get("figure_id"),
        })

    image_payloads = image_payloads[
        :MAX_IMAGES_TO_DISPLAY
    ]

    # Seules les images réellement envoyées à Qwen
    # sont ajoutées aux preuves autorisées.

    for image in image_payloads[:MAX_IMAGES_TO_VLM]:

        evidence.append({
            "id": image["id"],
            "page": image["page"],
            "type": "image",
            "content": (
                f"{image['resource_type']} — {image['description']}"
            ),
            "similarity": image["similarity"],
        })

    return evidence, image_payloads


# ============================================================
# 9. GENERATION LOCALE AVEC QWEN
# La fonction ask_qwen est centralisee dans maintidoc.generation.qwen.


# ============================================================
# 10. VALIDATION DES CITATIONS
# ============================================================

def validate_citations(
    response: dict,
    evidence: list[dict],
):

    allowed = {
        (item["id"], item["page"])
        for item in evidence
    }

    checked = []

    raw_citations = response.get("citations", [])

    if not isinstance(raw_citations, list):
        return [(raw_citations, False)]

    for citation in raw_citations:

        valid = (
            isinstance(citation, dict)
            and (
                citation.get("id"),
                citation.get("page"),
            ) in allowed
        )

        checked.append((citation, valid))

    return checked


# ============================================================
# 11. EN-TÊTE
# ============================================================

st.title("MaintiDoc · Assistant documentaire SAN")

st.caption(
    "Recherche documentaire multimodale avec "
    f"Qdrant et Qwen local ({LOCAL_LLM_MODEL})."
)

qwen_ready, qwen_status_detail = get_local_llm_status()
status_col, detail_col = st.columns([1, 3], vertical_alignment="center")
with status_col:
    st.badge(
        "Qwen prêt" if qwen_ready else "Qwen hors ligne",
        icon=":material/check_circle:" if qwen_ready else ":material/error:",
        color="green" if qwen_ready else "red",
    )
with detail_col:
    st.caption(
        f"Modèle local : {LOCAL_LLM_MODEL} · {qwen_status_detail}. "
        f"API : {LOCAL_LLM_BASE_URL}/models"
    )
st.caption("L’URL /v1 est une base d’API, pas une page web; une réponse 404 à sa racine est normale. MaintiDoc ne commande aucun équipement SAN.")


with st.sidebar:

    st.header("MaintiDoc")

    st.caption(
    f"{LOCAL_LLM_MODEL} · Qdrant"
    )

    with st.expander("État des services"):

        st.write(f"Modèle Qwen : `{LOCAL_LLM_MODEL}`")

        st.write(
            "Base vectorielle : Qdrant (vecteurs texte et images)"
        )

        st.write(
            "Embeddings texte : BGE-M3 (1024 dimensions)"
        )

        st.write(
            "Embeddings visuels : CLIP (512 dimensions)"
        )

        st.write(
            f"Images affichables : {MAX_IMAGES_TO_DISPLAY}"
        )

        st.write(
            f"Images envoyées à Qwen : {MAX_IMAGES_TO_VLM}"
        )


# ============================================================
# 13. INGESTION DES MANUELS PDF
# ============================================================

st.subheader("1. Ajouter un manuel")

upload = st.file_uploader(
    "Sélectionner un manuel PDF",
    type=["pdf"],
    key="manual_pdf",
)

if upload is not None:

    st.caption(
        f"Fichier : {upload.name} · {upload.size:,} octets"
    )

    if st.button(
        "Indexer ce manuel dans Qdrant",
        type="secondary",
    ):

        safe_filename = Path(upload.name).name

        if not safe_filename.lower().endswith(".pdf"):

            st.error("Le fichier sélectionné doit être un PDF.")

        else:

            try:

                with st.spinner(
                    "Extraction du PDF, génération des embeddings "
                    "et enregistrement dans Qdrant..."
                ):

                    # Conserver le nom original du PDF.
                    # ingest_pdf() récupère le nom depuis le chemin.
                    with tempfile.TemporaryDirectory(
                        prefix="maintidoc_"
                    ) as temp_dir:

                        temp_path = (
                            Path(temp_dir) / safe_filename
                        )

                        temp_path.write_bytes(
                            upload.getvalue()
                        )

                        result = ingest_pdf(
                            str(temp_path),
                            filename=safe_filename,
                        )

                # Le nom original est conservé dans l'état Streamlit.
                result["filename"] = safe_filename

                st.session_state["last_ingestion"] = result

                st.session_state["active_document_id"] = str(
                    result.get("document_id")
                )

                st.success(
                    "Le manuel a été indexé."
                )

                show_ingestion_metrics(result)

                with st.expander(
                    "Détails techniques de l'ingestion"
                ):
                    st.json(result)

            except Exception as exc:

                st.error(
                    f"Échec de l'ingestion : "
                    f"{type(exc).__name__}: {exc}"
                )

                st.exception(exc)


# ============================================================
# 14. RAPPEL DU DERNIER MANUEL INDEXÉ
# ============================================================

last_ingestion = st.session_state.get(
    "last_ingestion"
)

if last_ingestion:

    st.divider()

    st.markdown("### Dernier manuel indexé")

    st.write(
        f"**Nom :** {last_ingestion.get('filename', 'Inconnu')}"
    )

    st.write(
        f"**Identifiant :** "
        f"`{last_ingestion.get('document_id', 'Inconnu')}`"
    )

    show_ingestion_metrics(last_ingestion)


# ============================================================
# 15. INVENTAIRE DES MANUELS ENREGISTRÉS
# ============================================================

st.divider()

st.subheader("2. Manuels disponibles dans Qdrant")

try:

    from maintidoc.vector_store.qdrant_store import test_connection
    test_connection()
    database_documents = get_vector_documents()

    st.session_state["database_documents"] = (
        database_documents
    )

    if database_documents:

        failed_documents = [doc for doc in database_documents if (doc.get("metadata") or {}).get("index_status") == "failed"]
        ready_documents = [doc for doc in database_documents if (doc.get("metadata") or {}).get("index_status") != "failed"]

        st.success(
            f"{len(ready_documents)} manuel(s) trouvé(s) "
            "dans Qdrant."
        )

        if failed_documents:
            st.warning(
                "Certains PDF ont une ingestion inachevee et ne sont pas fiables pour la recherche. "
                "Relance leur ajout depuis Ajouter un manuel : "
                + ", ".join(doc.get("filename", "PDF sans nom") for doc in failed_documents)
            )

        with st.expander("Afficher l'inventaire"):

            st.dataframe(
                [
                    {
                        "Nom du manuel": doc["filename"],
                        "Pages": doc["page_count"],
                        "Identifiant": doc["document_id"],
                        "Indexation": (doc.get("metadata") or {}).get("index_status", "ancienne / a verifier"),
                    }
                    for doc in database_documents
                ],
                width="stretch",
                hide_index=True,
            )

    else:

        st.info(
            "Aucun manuel n'est actuellement enregistré dans Qdrant."
        )

except Exception as exc:

    database_documents = st.session_state.get(
        "database_documents",
        [],
    )

    st.warning(
        "Impossible d'actualiser l'inventaire Qdrant. Vérifie que le service "
        "Qdrant est démarré et que QDRANT_URL pointe vers son endpoint. "
        f"({type(exc).__name__}: {exc})"
    )


# ============================================================
# 16. INTERROGATION
# ============================================================

st.divider()

st.subheader("3. Poser une question")

question = st.text_area(
    "Question du technicien",
    placeholder=(
        "Ex. : Que vérifier en cas de perte "
        "d'un chemin Fibre Channel ?"
    ),
    height=100,
)
st.caption("Interrogez les manuels techniques indexés et vérifiez les références retournées.")

# ------------------------------------------------------------
# Choix du périmètre de recherche
# ------------------------------------------------------------

scope_options = [
    "Tous les manuels indexés"
]

if any((doc.get("metadata") or {}).get("index_status") != "failed" for doc in database_documents):
    scope_options.append("Choisir un manuel")

if st.session_state.get("active_document_id"):
    scope_options.append("Dernier manuel ajouté")

search_scope = st.radio(
    "Périmètre de recherche",
    scope_options,
    horizontal=True,
    label_visibility="collapsed",
)

selected_document_id = None

if search_scope == "Choisir un manuel":

    document_labels = {
        doc["document_id"]: (
            f"{doc['filename']} · "
            f"{doc['page_count'] or '?'} pages"
        )
        for doc in database_documents
    }

    selected_document_id = st.selectbox(
        "Sélectionner le manuel à interroger",
        options=[
            doc["document_id"]
            for doc in database_documents
        ],
        format_func=lambda document_id: (
            document_labels.get(
                document_id,
                document_id,
            )
        ),
        key="selected_document_for_search",
    )


# ============================================================
# 17. LANCEMENT DE LA RECHERCHE
# ============================================================

if st.button(
    "Rechercher et générer la réponse",
    type="primary",
    width="stretch",
):

    if not question.strip():

        st.warning(
            "Saisissez une question avant de lancer la recherche."
        )

    elif classify_question(question.strip())[0] != "document":
        st.info("Cette demande concerne un ticket ou un historique. Ouvrez la page Tickets dans le menu.")

    else:

        active_id = st.session_state.get(
            "active_document_id"
        )

        if search_scope == "Tous les manuels indexés":

            document_filter = None

        elif search_scope == "Choisir un manuel":

            document_filter = selected_document_id

        else:

            document_filter = active_id

        if (
            search_scope == "Dernier manuel ajouté"
            and not active_id
        ):

            st.warning(
                "Aucun manuel n'a été indexé pendant cette session."
            )

        elif (
            search_scope == "Choisir un manuel"
            and not document_filter
        ):

            st.warning(
                "Sélectionne un manuel avant de lancer la recherche."
            )

        else:

            # Éviter d'afficher une ancienne réponse.
            st.session_state.pop(
                "last_answer",
                None,
            )

            st.session_state.pop(
                "last_retrieval",
                None,
            )

            try:
                with st.spinner("LangGraph coordonne les agents MaintiDoc..."):
                    workflow = build_workflow(
                        retrieve_fn=retrieve,
                        prepare_fn=build_vlm_payload,
                        answer_fn=ask_qwen,
                        mcp_fn=call_maintidoc_tool,
                    )
                    result = workflow.invoke({
                        "question": question.strip(),
                        "document_id": document_filter,
                        "force_document": True,
                    })

                if result.get("intent") in {"document", "ticket_draft"}:
                    retrieved = result.get("retrieval", {})
                    evidence = result.get("evidence", [])
                    image_payloads = result.get("images", [])
                    raw = result.get("raw_answer", "")
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        parsed = None
                    st.session_state["last_retrieval"] = {
                        "question": question.strip(),
                        "document_filter": document_filter,
                        "retrieved": retrieved,
                        "evidence": evidence,
                        "selected_images": image_payloads,
                        "agent_trace": result.get("agent_trace", []),
                        "source": "draft" if result.get("intent") == "ticket_draft" else "document",
                        "draft_ticket": result.get("draft_ticket", {}),
                    }
                    answer_source = "draft" if result.get("intent") == "ticket_draft" else "document"
                else:
                    mcp_result = result.get("mcp_result", {})
                    if mcp_result.get("message"):
                        answer_text = mcp_result["message"]
                    elif isinstance(mcp_result.get("tickets"), list):
                        answer_text = f"{len(mcp_result['tickets'])} ticket(s) trouvé(s)."
                    elif isinstance(mcp_result.get("events"), list):
                        answer_text = f"{len(mcp_result['events'])} événement(s) trouvé(s)."
                    else:
                        answer_text = json.dumps(mcp_result, ensure_ascii=False, default=str)
                    raw = json.dumps({
                        "status": result.get("status", "mcp_answered"),
                        "answer": answer_text,
                        "citations": [],
                        "uncertainties": [],
                    }, ensure_ascii=False, default=str)
                    parsed = json.loads(raw)
                    st.session_state["last_retrieval"] = {
                        "question": question.strip(),
                        "document_filter": document_filter,
                        "retrieved": {"text_results": [], "image_results": [], "fused_results": []},
                        "evidence": [],
                        "selected_images": [],
                        "source": "mcp",
                        "mcp_result": mcp_result,
                        "equipment_id": result.get("equipment_id") or "",
                        "agent_trace": result.get("agent_trace", []),
                    }
                    answer_source = "mcp"

                st.session_state["last_answer"] = {
                    "raw": raw,
                    "parsed": parsed,
                    "source": answer_source,
                }

            except Exception as exc:

                st.error(
                    "Erreur pendant la recherche ou la generation : "
                    f"{type(exc).__name__}: {exc}"
                )


# ============================================================
# 18. AFFICHAGE DE LA RÉPONSE
# ============================================================

answer_data = st.session_state.get(
    "last_answer"
)

if answer_data:

    st.divider()

    st.subheader("4. Réponse de MaintiDoc")

    parsed = answer_data.get("parsed")
    raw = answer_data.get("raw", "")

    if parsed is None:

        st.warning(
            "Qwen n'a pas renvoyé un JSON valide. "
            "Voici sa réponse brute :"
        )

        st.code(raw)

    else:

        answer_text = parsed.get(
            "answer",
            "Aucune réponse fournie.",
        )

        st.caption(
            f"Statut : {parsed.get('status', 'inconnu')}"
        )
        if parsed.get("status") == "needs_clarification" and not st.session_state.get("last_retrieval", {}).get("evidence"):
            st.info(
                "Aucune preuve exploitable n'a ete retrouvee. Verifie dans l'inventaire que le manuel est marque 'indexed'. "
                "Si son ajout precedent a expire, relance l'ajout du meme PDF avant de poser la question."
            )

        agent_trace = st.session_state.get("last_retrieval", {}).get("agent_trace", [])
        if agent_trace:
            with st.expander("Agents LangGraph intervenus"):
                st.write(" → ".join(agent_trace))

        evidence = (
            st.session_state
            .get("last_retrieval", {})
            .get("evidence", [])
        )

        is_mcp_answer = answer_data.get("source") == "mcp"
        citations = [] if is_mcp_answer else validate_citations(parsed, evidence)

        if citations:
            st.markdown("#### Références annoncées")
            for citation, valid in citations:
                if valid:
                    st.write(f"- `{citation['id']}` · Page {citation['page']}")
                else:
                    st.warning(f"Citation non vérifiable : {citation}")
        elif not is_mcp_answer:
            st.info("Aucune citation structurée n'a été fournie.")

        invalid_citation = any(
            not valid for _, valid in citations
        )
        missing_answer_citation = (
            not is_mcp_answer
            and
            parsed.get("status") == "answered"
            and not citations
        )

        if not is_mcp_answer and (invalid_citation or missing_answer_citation):
            st.error(
                "La réponse est masquée car ses citations sont absentes "
                "ou ne correspondent pas aux preuves récupérées."
            )
        else:
            st.markdown(answer_text)

        uncertainties = parsed.get(
            "uncertainties",
            [],
        )

        if uncertainties:

            st.markdown("#### Points à vérifier")

            for uncertainty in uncertainties:

                st.warning(str(uncertainty))

        if not is_mcp_answer:
            st.caption(
                "La validation vérifie l'existence des IDs et des pages, "
                "mais ne garantit pas à elle seule la pertinence sémantique des citations."
            )

    st.download_button(
        "Télécharger la réponse",
        data=raw,
        file_name="maintidoc_reponse.json",
        mime="application/json",
    )


# ============================================================
# 19. DOCUMENTS ET PREUVES RETROUVÉS
# ============================================================

last = st.session_state.get(
    "last_retrieval"
)

if last:

    st.divider()

    st.subheader(
        "5. Documents et preuves retrouvés"
    )

    document_filter = last.get(
        "document_filter"
    )

    if document_filter:

        selected_doc = next(
            (
                doc
                for doc in st.session_state.get(
                    "database_documents",
                    [],
                )
                if doc["document_id"] == document_filter
            ),
            None,
        )

        if selected_doc:

            st.caption(
                f"Manuel interrogé : {selected_doc['filename']}"
            )

        else:

            st.caption(
                f"Identifiant du manuel : {document_filter}"
            )

    else:

        st.caption(
            "Périmètre : tous les manuels indexés dans Qdrant."
        )

    if last.get("source") == "mcp":
        st.caption("Résultat retourné par les outils MCP MaintiDoc.")
        mcp_result = last.get("mcp_result", {})
        rows = next((mcp_result.get(key) for key in ("tickets", "events") if isinstance(mcp_result.get(key), list)), None)
        if rows is not None:
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.info("Aucun résultat trouvé dans Qdrant pour cette demande.")
        elif isinstance(mcp_result.get("ticket"), dict):
            st.dataframe(pd.DataFrame([mcp_result["ticket"]]), hide_index=True, width="stretch")
        else:
            st.json(mcp_result)
        st.stop()

    retrieved = last["retrieved"]

    text_results = retrieved.get(
        "text_results",
        [],
    )

    image_results = retrieved.get(
        "image_results",
        [],
    )

    # --------------------------------------------------------
    # EXTRAITS TEXTUELS
    # --------------------------------------------------------

    table_results = [
        item
        for item in text_results
        if "table" in str(
            item.get("type")
            or item.get("chunk_type")
            or ""
        ).lower()
    ]

    ordinary_text_results = [
        item
        for item in text_results
        if item not in table_results
    ]

    st.markdown("### Extraits textuels")

    if ordinary_text_results:

        for index, item in enumerate(
            ordinary_text_results,
            start=1,
        ):

            page = item.get("page", "?")

            content = str(
                item.get("content") or ""
            )

            chunk_type = str(
                item.get("type")
                or item.get("chunk_type")
                or "texte"
            )

            similarity = item.get(
                "similarity"
            )

            with st.expander(
                f"Extrait {index} · Page {page} · {chunk_type}",
                expanded=False,
            ):

                metadata = item.get("metadata") or {}
                source_document = (
                    metadata.get("source_document")
                    or item.get("document_id")
                ) if isinstance(metadata, dict) else item.get("document_id")

                if source_document:
                    st.caption(f"Document : {source_document}")

                if similarity is not None:

                    st.caption(
                        f"Similarité : {to_plain(similarity)}"
                    )

                st.markdown(content)

    else:

        st.info(
            "Aucun extrait textuel retrouvé."
        )

    # --------------------------------------------------------
    # TABLEAUX
    # --------------------------------------------------------

    st.markdown("### Tableaux retrouvés")

    if table_results:

        for index, item in enumerate(
            table_results,
            start=1,
        ):

            st.markdown(
                f"**Tableau {index} · "
                f"Page {item.get('page', '?')}**"
            )

            metadata = item.get("metadata") or {}
            source_document = (
                metadata.get("source_document")
                or item.get("document_id")
            ) if isinstance(metadata, dict) else item.get("document_id")

            if source_document:
                st.caption(f"Document : {source_document}")

            table_rows = (
                metadata.get("table_rows", [])
                if isinstance(metadata, dict)
                else []
            )

            render_bordered_table(
                table_rows,
                str(item.get("content") or ""),
            )

    else:

        st.info(
            "Aucun segment identifié comme tableau n'a été retrouvé. "
            "Les tableaux doivent avoir été extraits et indexés "
            "pendant l'ingestion."
        )

    # --------------------------------------------------------
    # IMAGES RETROUVÉES
    # --------------------------------------------------------

    st.markdown(
        "### Images les mieux classées par CLIP"
    )

    if image_results:

        visible_images = image_results[
            :MAX_IMAGES_TO_DISPLAY
        ]

        columns = st.columns(2)

        displayed = 0

        for index, item in enumerate(visible_images):

            image_path = item.get(
                "image_path"
            )

            if not image_path:
                continue

            path = Path(str(image_path))

            if not path.is_file():
                continue

            page = item.get(
                "page",
                "?",
            )

            image_id = item.get(
                "image_id",
                f"Image {index + 1}",
            )

            description = item.get(
                "description"
            ) or "Aucune description disponible."

            metadata = item.get("metadata") or {}
            image_kind = (
                metadata.get("resource_type") or metadata.get("type")
                if isinstance(metadata, dict)
                else None
            ) or "original_image"
            image_kind_label = (
                "Rendu de page"
                if image_kind == "page_render"
                else "Image originale"
            )
            source_document = (
                metadata.get("source_pdf_filename")
                or item.get("document_id")
                or "Document inconnu"
            ) if isinstance(metadata, dict) else item.get("document_id", "Document inconnu")

            try:

                similarity = float(
                    item.get("similarity")
                )

                similarity_label = (
                    f"Similarité CLIP : {similarity:.4f}"
                )

            except (TypeError, ValueError):

                similarity_label = (
                    "Score de similarité indisponible"
                )

            with columns[displayed % 2]:

                st.image(
                    str(path),
                    caption=f"{image_kind_label} · {image_id} · Page {page} · {source_document}",
                    width="stretch",
                )

                st.caption(str(description))

                st.caption(similarity_label)

            displayed += 1

        if displayed == 0:

            st.warning(
                "Des résultats d'images existent dans Qdrant, "
                "mais leurs fichiers ne sont pas accessibles "
                "depuis ce poste."
            )

    else:

        st.info(
            "Aucune image retrouvée par la recherche visuelle."
        )

    # --------------------------------------------------------
    # IMAGES TRANSMISES À QWEN
    # --------------------------------------------------------

    st.markdown(
        "### Images effectivement transmises à Qwen"
    )

    selected_images = last.get(
        "selected_images",
        [],
    )

    if selected_images:

        for image in selected_images[:MAX_IMAGES_TO_VLM]:

            st.write(
                f"- `{image['id']}` · "
                f"Page {image['page']} · "
                f"{image['resource_type']} · "
                f"{image.get('source_document') or 'Document inconnu'} · "
                f"{image['mime_type']}"
            )

    else:

        st.info(
            "Aucune image n'a été transmise à Qwen."
        )

    # --------------------------------------------------------
    # PREUVES TRANSMISES À QWEN
    # --------------------------------------------------------

    st.markdown(
        "### Preuves transmises à Qwen"
    )

    for item in last["evidence"]:

        st.write(
            f"- {item['type']} · "
            f"ID : `{item['id']}` · "
            f"Page : {item['page']}"
        )


# ============================================================
# 20. PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "MaintiDoc est un prototype pédagogique. "
    "Les réponses doivent être vérifiées par un technicien habilité. "
    "Aucune commande sur le SAN et aucune création automatique "
    "de ticket ne sont exécutées."
)
