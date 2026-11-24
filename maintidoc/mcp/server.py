"""Serveur MCP local pour l'historique et les brouillons MaintiDoc."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maintidoc.vector_store.qdrant_store import create_ticket_record, list_records

mcp = MCPServer(
    "MaintiDoc SAN",
    instructions=(
        "Outils Qdrant MaintiDoc. Aucune commande SAN. "
        "create_ticket enregistre un brouillon local apres confirmation humaine."
    ),
)


def _equipment(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 100:
        raise ValueError("equipment_id doit contenir entre 1 et 100 caracteres.")
    return value


@mcp.tool()
def get_equipment_history(equipment_id: str) -> dict[str, Any]:
    """Renvoie l'historique d'un equipement enregistre dans Qdrant."""
    equipment_id = _equipment(equipment_id)
    events = list_records("history", equipment_id=equipment_id)
    events.sort(key=lambda item: item.get("event_date") or item.get("created_at") or "", reverse=True)
    return {"equipment_id": equipment_id, "events": events[:100]}


@mcp.tool()
def get_open_tickets(equipment_id: str | None = None) -> dict[str, Any]:
    """Renvoie les brouillons et tickets actifs de Qdrant."""
    if equipment_id:
        equipment_id = _equipment(equipment_id)
    tickets = list_records("ticket", equipment_id=equipment_id)
    tickets = [item for item in tickets if item.get("status") in {"draft", "pending_approval", "approved", "created"}]
    tickets.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"tickets": tickets[:100]}


@mcp.tool()
def create_ticket(
    equipment_id: str,
    title: str,
    description: str,
    human_confirmed: bool,
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Enregistre un brouillon local apres confirmation humaine; aucun ticket externe."""
    if human_confirmed is not True:
        raise PermissionError("Une confirmation humaine explicite est requise.")
    equipment_id = _equipment(equipment_id)
    title, description = title.strip(), description.strip()
    if not title or len(title) > 500 or not description or len(description) > 10000:
        raise ValueError("Titre requis (500 caracteres max) et description requise (10000 max).")
    key = idempotency_key.strip() or hashlib.sha256(f"{equipment_id}|{title}|{description}".encode()).hexdigest()
    if len(key) > 255:
        raise ValueError("idempotency_key ne peut depasser 255 caracteres.")
    ticket, created = create_ticket_record(equipment_id, title, description, key)
    message = "Brouillon enregistre dans Qdrant. Approbation humaine requise; aucun ticket externe n'a ete cree."
    if not created:
        message = "Ce brouillon existait deja (cle d'idempotence); aucun doublon n'a ete cree."
    return {"ticket": ticket, "message": message}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
