"""Orchestration MaintiDoc par agents spécialisés LangGraph.

Agents documentaires : intent, retrieval (preuves) et response (reponse et citations).
Les demandes opérationnelles passent par l'agent MCP; aucun outil d'écriture
ticket n'est appelé avant la confirmation explicite dans l'interface.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class MaintiDocState(TypedDict, total=False):
    question: str
    force_document: bool
    search_query: str
    document_id: str | None
    intent: str
    mcp_action: str
    equipment_id: str | None
    mcp_result: dict[str, Any]
    retrieval: dict[str, Any]
    evidence: list[dict[str, Any]]
    images: list[dict[str, Any]]
    raw_answer: str
    status: str
    attempts: int
    agent_trace: list[str]
    citation_validation: dict[str, Any]
    draft_ticket: dict[str, Any]
    ticket_description: str
    ticket_alarm_code: str
    ticket_title: str


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in value if not unicodedata.combining(char)).casefold()


def classify_question(question: str) -> tuple[str, str | None, str | None]:
    """Route les questions documentaires et les demandes MCP explicites."""
    normalized = _normalize(question).strip()
    equipment = re.search(r"\bSAN[-_ ]?LAB[-_ ]?\d{3,4}\b|\bSANLAB\d{3,4}\b", question, re.I)
    equipment_id = re.sub(r"[-_ ]+", "-", equipment.group(0)).upper() if equipment else None

    # Les formulations interrogatives restent en RAG même si elles contiennent
    # un verbe qui peut également exprimer une action.
    documentary_scope = (
        normalized.startswith(("comment ", "explique comment ", "quelle procedure", "quelles sont les etapes"))
        or any(phrase in normalized for phrase in ("dans le manuel", "selon le manuel", "dans le document", "dans le pdf", "workflow"))
    )
    if documentary_scope:
        return "document", None, equipment_id

    create_action = re.search(
        r"\b(creer|cree|creez|ouvrir|ouvre|ouvrez|preparer|prepare|preparez|rediger|redige|redigez)\b.{0,50}\bticket\b",
        normalized,
    )
    if create_action:
        return "ticket_draft", "create_ticket", equipment_id

    if any(word in normalized for word in ("historique", "antecedent", "interventions passees", "evenements passes")):
        return "equipment_history", "get_equipment_history", equipment_id

    has_ticket = re.search(r"\btickets?\b", normalized) is not None
    asks_to_retrieve = any(word in normalized for word in (
        "donne", "donner", "montre", "affiche", "liste", "retrouve", "consulte",
        "quel", "quels", "ouverte", "ouvert", "en cours", "existant", "statut", "attente", "pending",
    ))
    if has_ticket and asks_to_retrieve:
        return "open_tickets", "get_open_tickets", equipment_id
    return "document", None, equipment_id


def _validate_answer_citations(raw_answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Valide le JSON et l'existence exacte des couples ID/page cités."""
    try:
        parsed = json.loads(raw_answer)
    except (TypeError, json.JSONDecodeError):
        return {"valid": False, "reason": "invalid_json", "citations": []}
    allowed = {
        (str(item.get("id")), int(item.get("page") or 0))
        for item in evidence
        if item.get("id") is not None
    }
    citations = parsed.get("citations", [])
    if not isinstance(citations, list):
        return {"valid": False, "reason": "citations_not_a_list", "citations": []}
    invalid = []
    for citation in citations:
        if not isinstance(citation, dict):
            invalid.append(citation)
            continue
        try:
            pair = (str(citation.get("id")), int(citation.get("page")))
        except (TypeError, ValueError):
            invalid.append(citation)
            continue
        if pair not in allowed:
            invalid.append(citation)
    if parsed.get("status") == "answered" and not citations:
        invalid.append("missing_citations")
    return {
        "valid": not invalid,
        "reason": "ok" if not invalid else "unknown_or_missing_citations",
        "citations": citations,
        "invalid": invalid,
        "answer": parsed,
    }


def build_workflow(
    retrieve_fn: Callable[..., dict[str, Any]] | None = None,
    prepare_fn: Callable[[dict[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    answer_fn: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str] | None = None,
    mcp_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
):
    """Construit un graphe compact de six roles specialisees."""

    def trace(state: MaintiDocState, agent_name: str) -> list[str]:
        return [*state.get("agent_trace", []), agent_name]

    def intent_agent(state: MaintiDocState) -> dict[str, Any]:
        intent, action, equipment_id = classify_question(state["question"])
        if state.get("force_document"):
            intent, action = "document", None
        return {
            "intent": intent,
            "mcp_action": action or "",
            "equipment_id": equipment_id,
            "search_query": state["question"],
            "attempts": 0,
            "agent_trace": trace(state, "intent_agent"),
        }

    def retrieval_agent(state: MaintiDocState) -> dict[str, Any]:
        if retrieve_fn is None or prepare_fn is None:
            raise RuntimeError("Le parcours documentaire requiert les fonctions de recherche et de préparation des preuves.")
        query = state.get("search_query") or state["question"]
        result = retrieve_fn(query, document_id=state.get("document_id"))
        evidence, images = prepare_fn(result)
        evidence = [item for item in evidence if item.get("id") is not None and int(item.get("page") or 0) > 0]
        return {
            "retrieval": result,
            "evidence": evidence,
            "images": images,
            "attempts": state.get("attempts", 0) + 1,
            "agent_trace": trace(state, "retrieval_agent"),
        }

    def response_agent(state: MaintiDocState) -> dict[str, Any]:
        if not state.get("evidence"):
            answer = {
                "status": "needs_clarification",
                "answer": "Je n'ai pas retrouvé de preuve documentaire suffisante. Précise le code, le composant ou le manuel concerné.",
                "citations": [],
                "uncertainties": ["Aucune preuve exploitable n'a été retrouvée."],
            }
            raw = json.dumps(answer, ensure_ascii=False)
        else:
            if answer_fn is None:
                raise RuntimeError("Le parcours documentaire requiert le modèle Qwen local.")
            raw = answer_fn(state["question"], state["evidence"], state.get("images", []))
        validation = _validate_answer_citations(raw, state.get("evidence", []))
        parsed = validation.get("answer")
        if parsed is not None and not validation["valid"] and parsed.get("status") == "answered":
            parsed["status"] = "needs_clarification"
            parsed["answer"] = "La réponse a été retenue car ses citations ne correspondent pas exactement aux preuves récupérées."
            parsed["uncertainties"] = [*parsed.get("uncertainties", []), validation["reason"]]
            raw = json.dumps(parsed, ensure_ascii=False)
        return {"raw_answer": raw, "citation_validation": validation, "agent_trace": trace(state, "response_agent")}

    def ticket_draft_agent(state: MaintiDocState) -> dict[str, Any]:
        """Met en forme les seuls champs du ticket, sans RAG, Qwen ni écriture."""
        equipment_id = state.get("equipment_id")
        issue = str(state.get("ticket_description") or "").strip()
        if not equipment_id or not issue:
            return {
                "status": "needs_clarification",
                "raw_answer": json.dumps({
                    "status": "needs_clarification",
                    "answer": "Renseigne l'identifiant de l'équipement et le problème observé.",
                    "citations": [],
                    "uncertainties": [],
                }, ensure_ascii=False),
                "agent_trace": trace(state, "ticket_draft_agent"),
            }

        alarm_code = str(state.get("ticket_alarm_code") or "").strip().upper()
        title = str(state.get("ticket_title") or "").strip()
        if not title:
            topic = alarm_code if alarm_code else "maintenance"
            title = f"Signalement {topic} — {equipment_id}"

        description_parts = []
        if alarm_code:
            description_parts.append(f"Code d'alarme déclaré : {alarm_code}")
        description_parts.append(f"Description du technicien : {issue}")
        description_parts.append(
            "Brouillon préparé à partir des informations saisies; "
            "aucun manuel ni système SAN n'a été interrogé."
        )
        draft = {
            "equipment_id": equipment_id,
            "title": title,
            "description": "\n\n".join(description_parts),
        }
        answer = {
            "status": "draft_confirmation_required",
            "answer": "Brouillon local prêt à vérifier. Aucune donnée n'a été enregistrée.",
            "citations": [],
            "uncertainties": ["Le signalement reprend les informations déclarées par le technicien."],
        }
        return {
            "status": "draft_confirmation_required",
            "draft_ticket": draft,
            "raw_answer": json.dumps(answer, ensure_ascii=False),
            "agent_trace": trace(state, "ticket_draft_agent"),
        }

    def draft_confirmation_agent(state: MaintiDocState) -> dict[str, Any]:
        draft = state.get("draft_ticket") or {}
        if not all(str(draft.get(field) or "").strip() for field in ("equipment_id", "title", "description")):
            return {
                "status": "needs_clarification",
                "agent_trace": trace(state, "draft_confirmation_agent"),
            }
        return {
            "status": "draft_confirmation_required",
            "draft_ticket": draft,
            "agent_trace": trace(state, "draft_confirmation_agent"),
        }

    def mcp_agent(state: MaintiDocState) -> dict[str, Any]:
        action = state.get("mcp_action", "")
        if mcp_fn is None:
            return {"status": "mcp_unavailable", "mcp_result": {"message": "Serveur MCP non configuré ou indisponible."}, "agent_trace": trace(state, "mcp_agent")}
        if action == "get_equipment_history" and not state.get("equipment_id"):
            return {"status": "needs_clarification", "mcp_result": {"message": "Précise l'identifiant de l'équipement (ex. SAN-LAB-2000)."}, "agent_trace": trace(state, "mcp_agent")}
        try:
            args = {"equipment_id": state.get("equipment_id")} if state.get("equipment_id") else {}
            return {"status": "mcp_answered", "mcp_result": mcp_fn(action, args), "agent_trace": trace(state, "mcp_agent")}
        except Exception as exc:
            return {"status": "mcp_unavailable", "mcp_result": {"message": f"Appel MCP impossible : {type(exc).__name__}: {exc}"}, "agent_trace": trace(state, "mcp_agent")}

    def route_after_intent(state: MaintiDocState) -> str:
        if state.get("intent") in {"open_tickets", "equipment_history"}:
            return "mcp"
        if state.get("intent") == "ticket_draft":
            return "draft"
        return "retrieve"

    graph = StateGraph(MaintiDocState)
    graph.add_node("intent_agent", intent_agent)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("response_agent", response_agent)
    graph.add_node("ticket_draft_agent", ticket_draft_agent)
    graph.add_node("draft_confirmation_agent", draft_confirmation_agent)
    graph.add_node("mcp_agent", mcp_agent)
    graph.add_edge(START, "intent_agent")
    graph.add_conditional_edges(
        "intent_agent",
        route_after_intent,
        {"mcp": "mcp_agent", "draft": "ticket_draft_agent", "retrieve": "retrieval_agent"},
    )
    graph.add_edge("retrieval_agent", "response_agent")
    graph.add_edge("response_agent", END)
    graph.add_edge("ticket_draft_agent", "draft_confirmation_agent")
    graph.add_edge("draft_confirmation_agent", END)
    graph.add_edge("mcp_agent", END)
    return graph.compile()
