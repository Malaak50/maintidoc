"""Stockage MaintiDoc dans Qdrant (chunks, figures et donnees de suivi)."""
from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DOCUMENTS = "maintidoc_documents"
CHUNKS = "maintidoc_chunks"
IMAGES = "maintidoc_images"
RECORDS = "maintidoc_records"
COLLECTIONS = {
    DOCUMENTS: 1,
    CHUNKS: 1024,
    IMAGES: 512,
    RECORDS: 1,
}
_NAMESPACE = uuid.UUID("3ae0d5be-e259-4fe3-9c22-78dfd80c4213")
_client = None


def get_client():
    """Retourne le client Qdrant serveur configure par QDRANT_URL."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").strip()
        api_key = os.getenv("QDRANT_API_KEY") or None
        timeout = float(os.getenv("QDRANT_TIMEOUT", "140"))
        _client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
    return _client


def _ensure_collections(client=None):
    from qdrant_client.http.models import Distance, VectorParams

    client = client or get_client()
    for name, size in COLLECTIONS.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
    return client


def test_connection():
    """Verifie Qdrant et initialise les collections applicatives."""
    client = _ensure_collections()
    client.get_collections()
    return {"backend": "qdrant", "collections": sorted(COLLECTIONS), "status": "ok"}


def _point_id(kind: str, key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{key}"))


def _filter(**values):
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in values.items()])


def _scroll(collection: str, query_filter=None, limit: int = 1000):
    client = _ensure_collections()
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=min(256, max(1, limit - len(points))),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None or len(points) >= limit:
            break
    return points


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Le payload Qdrant est JSON: convertir les objets Path et les valeurs numpy.
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
    return convert(payload)


def save_document(document_id: str, filename: str, sha256: str, revision=None, page_count=None, metadata=None, connection=None):
    from qdrant_client.http.models import PointStruct
    if not document_id or not document_id.strip() or not filename or not filename.strip():
        raise ValueError("document_id et filename sont obligatoires.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256 or ""):
        raise ValueError("Le SHA-256 doit contenir 64 caracteres hexadecimaux.")
    payload = {
        "entity_type": "document", "document_id": document_id, "filename": filename,
        "sha256": sha256.lower(), "revision": revision, "page_count": page_count,
        "metadata": {**_safe_payload(metadata or {}), "index_status": "indexing"},
    }
    _ensure_collections().upsert(
        collection_name=DOCUMENTS,
        points=[PointStruct(id=_point_id("document", document_id), vector=[0.0], payload=payload)],
        wait=True,
    )
    return document_id


def set_document_index_status(document_id: str, status: str, error: str | None = None) -> None:
    """Marque un document comme indexe ou incomplet apres une ingestion."""
    from qdrant_client.http.models import PointStruct

    if status not in {"indexing", "indexed", "failed"}:
        raise ValueError("Statut d'indexation invalide.")
    client = _ensure_collections()
    point_id = _point_id("document", document_id)
    existing = client.retrieve(
        collection_name=DOCUMENTS,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not existing:
        raise ValueError(f"Document introuvable dans Qdrant : {document_id}")
    payload = dict(existing[0].payload or {})
    metadata = dict(payload.get("metadata") or {})
    metadata["index_status"] = status
    if error:
        metadata["index_error"] = error[:1000]
    else:
        metadata.pop("index_error", None)
    payload["metadata"] = metadata
    client.upsert(
        collection_name=DOCUMENTS,
        points=[PointStruct(id=point_id, vector=[0.0], payload=payload)],
        wait=True,
    )


def save_chunks(document_id: str, chunks: list[dict], connection=None) -> int:
    from qdrant_client.http.models import PointStruct

    if not isinstance(chunks, list):
        raise TypeError("chunks doit etre une liste.")
    client = _ensure_collections()
    points = []
    seen = set()
    for chunk in chunks:
        chunk_id, content = chunk.get("id"), chunk.get("text")
        page, index = chunk.get("page"), chunk.get("chunk_index")
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen:
            raise ValueError("Identifiant de chunk invalide ou duplique.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Contenu vide pour {chunk_id}.")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"Numero de page invalide pour {chunk_id}.")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError(f"Index invalide pour {chunk_id}.")
        seen.add(chunk_id)
        kind = chunk.get("type", "text")
        metadata = {
            "source_id": chunk.get("source_id"), "source_document": chunk.get("document"),
            "source_sha256": chunk.get("sha256"), "original_type": kind,
        }
        for field in ("ocr_attempted", "extraction_method"):
            if field in chunk:
                metadata[field] = chunk[field]
        if kind.lower() in {"table", "tableau"}:
            metadata["table_rows"] = chunk.get("table_rows", [])
        payload = {
            "chunk_id": chunk_id, "document_id": document_id, "page": page,
            "chunk_index": index, "chunk_type": kind, "content": content.strip(),
            "metadata": _safe_payload(metadata), "embedding_ready": False,
        }
        points.append(PointStruct(id=_point_id("chunk", chunk_id), vector=[0.0] * 1024, payload=payload))
    if points:
        existing = client.retrieve(
            collection_name=CHUNKS,
            ids=[point.id for point in points],
            with_payload=True,
            with_vectors=False,
        )
        for point in existing:
            if (point.payload or {}).get("document_id") != document_id:
                raise ValueError(
                    f"Le chunk {(point.payload or {}).get('chunk_id')} appartient deja a un autre document."
                )
        client.upsert(collection_name=CHUNKS, points=points, wait=True)
    return len(points)


def save_embeddings(document_id: str, chunk_ids: list[str], embeddings, connection=None) -> int:
    from qdrant_client.http.models import PointStruct

    if len(chunk_ids) != len(embeddings):
        raise ValueError("Le nombre d'embeddings doit correspondre aux chunks.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Les identifiants doivent etre uniques.")
    client = _ensure_collections()
    points = []
    for chunk_id, vector in zip(chunk_ids, embeddings):
        values = [float(value) for value in vector]
        if len(values) != 1024 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Embedding invalide (1024 dimensions attendues): {chunk_id}")
        points.append((chunk_id, _point_id("chunk", chunk_id), values))
    if points:
        existing = client.retrieve(
            collection_name=CHUNKS,
            ids=[point_id for _, point_id, _ in points],
            with_payload=True,
            with_vectors=False,
        )
        by_id = {str(point.id): point for point in existing}
        upserts = []
        for chunk_id, point_id, values in points:
            stored = by_id.get(str(point_id))
            if stored is None or (stored.payload or {}).get("document_id") != document_id:
                raise ValueError(f"Chunk introuvable ou non unique: {chunk_id}")
            upserts.append(PointStruct(
                id=stored.id,
                vector=values,
                payload={**(stored.payload or {}), "embedding_ready": True},
            ))
        client.upsert(collection_name=CHUNKS, points=upserts, wait=True)
    return len(points)


def save_image_embeddings(document_id: str, image_records: list[dict], embeddings, connection=None) -> int:
    from qdrant_client.http.models import PointStruct

    if len(image_records) != len(embeddings):
        raise ValueError("Le nombre d'embeddings doit correspondre aux images.")
    points = []
    seen = set()
    for record, vector in zip(image_records, embeddings):
        image_id = record.get("id")
        path = Path(record.get("path", ""))
        values = [float(value) for value in vector]
        if not isinstance(image_id, str) or not image_id.strip() or image_id in seen:
            raise ValueError("Identifiant d'image invalide ou duplique.")
        if record.get("document_id") != document_id or not path.is_file():
            raise ValueError(f"Document ou fichier image invalide pour {image_id}.")
        if len(values) != 512 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Embedding CLIP invalide pour {image_id}.")
        seen.add(image_id)
        metadata = _safe_payload(record.get("metadata") or {})
        image_type = record.get("type", "image")
        payload = {
            "image_id": image_id, "document_id": document_id, "page": int(record["page"]),
            "filename": record.get("filename"), "image_path": str(path), "width": int(record["width"]),
            "height": int(record["height"]), "figure_id": record.get("figure_id"),
            "description": record.get("description"), "resource_type": metadata.get("resource_type", image_type),
            "metadata": {**metadata, "source_pdf_filename": record.get("filename"), "type": image_type, "sha256": record.get("sha256")},
        }
        points.append(PointStruct(id=_point_id("image", image_id), vector=values, payload=_safe_payload(payload)))
    if points:
        _ensure_collections().upsert(collection_name=IMAGES, points=points, wait=True)
    return len(points)


def _to_chunk(point, score=0.0):
    payload = dict(point.payload or {})
    return {**payload, "similarity": float(score)}


def search_chunks(query_embedding, top_k=5, document_id=None):
    values = [float(value) for value in query_embedding]
    if len(values) != 1024 or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding textuel invalide: 1024 dimensions attendues.")
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    filters = [FieldCondition(key="embedding_ready", match=MatchValue(value=True))]
    if document_id:
        filters.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    result = _ensure_collections().query_points(
        collection_name=CHUNKS, query=values, query_filter=Filter(must=filters),
        limit=top_k, with_payload=True,
    )
    return [_to_chunk(point, point.score) for point in result.points]


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def search_chunks_lexical(query: str, top_k=10, document_id=None):
    stopwords = {"que", "quel", "quelle", "quels", "quelles", "est", "sont", "dans", "pour", "avec", "sur", "les", "des", "une", "un", "du", "de", "la", "le", "et", "comment", "donne", "donner", "moi", "this", "that", "what", "which", "show", "list", "the", "for", "with", "from", "are", "was", "were", "how"}
    words = [word for word in re.findall(r"[\w-]+", _normalize(query)) if len(word) > 2 and word not in stopwords]
    if not words:
        return []
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    query_filter = Filter(must=[FieldCondition(key="embedding_ready", match=MatchValue(value=True))])
    if document_id:
        query_filter.must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    scored = []
    for point in _scroll(CHUNKS, query_filter, limit=100000):
        item = _to_chunk(point)
        content = _normalize(str(item.get("content") or ""))
        lexical_score = sum(content.count(word) for word in words)
        if item.get("chunk_type") == "table" and any(word in {"ticket", "tickets", "alarme", "alarmes", "code", "codes", "tableau", "tableaux"} for word in words):
            lexical_score += 2
        if lexical_score:
            item["lexical_score"] = lexical_score
            scored.append(item)
    return sorted(scored, key=lambda item: (-item["lexical_score"], item.get("page", 0), item.get("chunk_index", 0)))[:top_k]


def search_images(query_embedding, top_k=5, document_id=None):
    values = [float(value) for value in query_embedding]
    if len(values) != 512 or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding CLIP invalide: 512 dimensions attendues.")
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    must = []
    must_not = [FieldCondition(key="resource_type", match=MatchValue(value="page_render"))]
    if document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    query_filter = Filter(must=must or None, must_not=must_not)
    result = _ensure_collections().query_points(collection_name=IMAGES, query=values, query_filter=query_filter, limit=top_k, with_payload=True)
    return [{**dict(point.payload or {}), "similarity": float(point.score)} for point in result.points]


def list_documents():
    points = _scroll(DOCUMENTS, limit=100000)
    rows = [point.payload for point in points if point.payload and point.payload.get("entity_type") == "document"]
    return sorted(rows, key=lambda row: (row.get("filename") or "", row.get("document_id") or ""))


def upsert_record(entity_type: str, record_id: str, payload: dict[str, Any]):
    from qdrant_client.http.models import PointStruct
    body = {**_safe_payload(payload), "entity_type": entity_type, "record_id": record_id}
    _ensure_collections().upsert(collection_name=RECORDS, points=[PointStruct(id=_point_id(entity_type, record_id), vector=[0.0], payload=body)], wait=True)
    return body


def list_records(entity_type: str, equipment_id: str | None = None, limit=100):
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    must = [FieldCondition(key="entity_type", match=MatchValue(value=entity_type))]
    if equipment_id:
        must.append(FieldCondition(key="equipment_id", match=MatchValue(value=equipment_id)))
    return [point.payload for point in _scroll(RECORDS, Filter(must=must), limit=limit) if point.payload]


def create_ticket_record(equipment_id: str, title: str, description: str, idempotency_key: str):
    key = idempotency_key or hashlib.sha256(f"{equipment_id}|{title}|{description}".encode()).hexdigest()
    record_id = key
    existing = _scroll(RECORDS, _filter(entity_type="ticket", idempotency_key=key), limit=2)
    if existing:
        ticket = existing[0].payload
        return ticket, False
    from datetime import datetime, timezone
    ticket = {
        "ticket_id": _point_id("ticket-number", key), "external_id": None,
        "equipment_id": equipment_id, "title": title, "description": description,
        "status": "draft", "created_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": key, "human_approval_required": True,
    }
    upsert_record("ticket", record_id, ticket)
    upsert_record("audit", key, {
        "operation": "create_draft", "entity_type": "maintenance_ticket",
        "entity_id": ticket["ticket_id"], "actor": "streamlit_human_confirmed",
        "details": {"equipment_id": equipment_id, "idempotency_key": key, "human_approval_required": True},
        "created_at": ticket["created_at"],
    })
    return ticket, True
