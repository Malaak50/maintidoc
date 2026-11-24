"""Espace autonome de pr?paration et consultation des tickets MaintiDoc."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from maintidoc.agent.workflow import build_workflow
from maintidoc.mcp.client import call_maintidoc_tool

st.title("Tickets et interventions")
st.caption("Preparez un brouillon a partir de vos observations, consultez les tickets enregistres ou recherchez les interventions consignees.")
mode = st.segmented_control(
    "Espace de travail",
    ["Preparer un brouillon", "Consulter les tickets", "Historique des interventions"],
    default="Consulter les tickets",
    key="tickets_page_mode",
    width="stretch",
)

if mode == "Preparer un brouillon":
    st.subheader("Nouveau signalement")
    st.caption("Le brouillon est prepare localement a partir de votre saisie. Il ne lance ni recherche documentaire ni modele Qwen.")
    with st.form("ticket_draft_form", border=True):
        equipment = st.text_input("Equipement", value="SAN-LAB-2000", key="ticket_equipment_input")
        alarm = st.text_input("Code alarme (facultatif)", key="ticket_alarm_input")
        issue = st.text_area("Probleme observe", placeholder="Decrivez les symptomes, le composant et le moment d'apparition.", height=130, key="ticket_issue_input")
        submitted = st.form_submit_button("Preparer le brouillon", type="primary", icon=":material/description:", width="stretch")
    if submitted:
        if not equipment.strip() or not issue.strip():
            st.warning("L'Equipement et la description sont obligatoires.")
        else:
            graph = build_workflow(mcp_fn=call_maintidoc_tool)
            result = graph.invoke({"question": f"Preparer un brouillon de ticket pour {equipment.strip()}", "ticket_description": issue.strip(), "ticket_alarm_code": alarm.strip()})
            if result.get("draft_ticket"):
                for key in ("pending_ticket_equipment", "pending_ticket_title", "pending_ticket_description", "pending_ticket_human_confirmation"):
                    st.session_state.pop(key, None)
                st.session_state["pending_ticket_draft"] = {"ticket": result["draft_ticket"], "agent_trace": result.get("agent_trace", [])}
            else:
                st.warning(json.loads(result.get("raw_answer", "{}")).get("answer", "Completez les champs requis."))

    pending = st.session_state.get("pending_ticket_draft")
    if pending:
        proposed = pending["ticket"]
        with st.container(border=True):
            st.subheader("Relire et confirmer")
            st.caption("Aucune donnee n'est enregistree avant la confirmation ci-dessous.")
            st.caption("Etapes LangGraph : " + " -> ".join(pending.get("agent_trace", [])))
            with st.form("confirm_local_ticket_draft", border=False):
                confirmed_equipment = st.text_input("Equipement", value=proposed.get("equipment_id", equipment if submitted else ""), key="pending_ticket_equipment")
                title = st.text_input("Titre", value=proposed.get("title", "Signalement maintenance"), key="pending_ticket_title")
                description = st.text_area("Description modifiable", value=proposed.get("description", ""), height=150, key="pending_ticket_description")
                confirmed = st.checkbox("Je confirme l'enregistrement de ce brouillon local.", key="pending_ticket_human_confirmation")
                save = st.form_submit_button("Confirmer et enregistrer", type="primary", icon=":material/save:")
            if save:
                if not confirmed:
                    st.warning("Cochez la confirmation avant l'enregistrement.")
                elif not all(v.strip() for v in (confirmed_equipment, title, description)):
                    st.warning("Tous les champs sont obligatoires.")
                else:
                    try:
                        idem = hashlib.sha256(f"{confirmed_equipment.strip()}|{title.strip()}|{description.strip()}".encode()).hexdigest()
                        saved = call_maintidoc_tool("create_ticket", {"equipment_id": confirmed_equipment.strip(), "title": title.strip(), "description": description.strip(), "human_confirmed": True, "idempotency_key": idem})
                        st.session_state.pop("pending_ticket_draft", None)
                        st.success(saved.get("message", "Brouillon enregistre dans Qdrant."))
                        st.json(saved.get("ticket", {}))
                    except Exception as exc:
                        st.error(f"Enregistrement MCP impossible : {type(exc).__name__}: {exc}")
else:
    is_tickets = mode == "Consulter les tickets"
    st.subheader("Tickets enregistres" if is_tickets else "Historique des interventions")
    with st.form("ticket_read_form", border=True):
        equipment_filter = st.text_input("Filtrer par Equipement (facultatif)", value="SAN-LAB-2000", key="ticket_read_equipment")
        read = st.form_submit_button("Afficher", type="primary", icon=":material/search:")
    if read:
        args = {"equipment_id": equipment_filter.strip()} if equipment_filter.strip() else {}
        if is_tickets:
            try:
                st.session_state["ticket_read_result"] = {
                    "mode": mode,
                    "payload": call_maintidoc_tool("get_open_tickets", args),
                }
            except Exception as exc:
                st.session_state["ticket_read_result"] = {
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            payload = {}
            errors = []
            for key, tool, tool_args in (
                ("tickets", "get_open_tickets", args),
                ("history", "get_equipment_history", args),
            ):
                try:
                    payload[key] = call_maintidoc_tool(tool, tool_args)
                except Exception as exc:
                    errors.append(f"{tool}: {type(exc).__name__}: {exc}")
            st.session_state["ticket_read_result"] = {
                "mode": mode,
                "payload": payload,
                "errors": errors,
            }
    result = st.session_state.get("ticket_read_result", {})
    if result.get("mode") == mode:
        if result.get("error"):
            st.error(f"Lecture MCP impossible : {result['error']}")
        else:
            payload = result.get("payload", {})
            if is_tickets:
                rows = payload.get("tickets")
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.info("Aucun ticket enregistre pour cet equipement.")
            else:
                ticket_result = payload.get("tickets", {})
                history_result = payload.get("history", {})
                if result.get("errors"):
                    st.warning("Certaines donnees MCP n'ont pas pu etre lues : " + " | ".join(result["errors"]))

                st.markdown("#### Brouillons et tickets enregistres")
                tickets = ticket_result.get("tickets", [])
                if tickets:
                    st.dataframe(pd.DataFrame(tickets), hide_index=True, width="stretch")
                else:
                    st.info("Aucun brouillon ou ticket enregistre pour cet equipement.")

                st.markdown("#### Interventions consignees")
                events = history_result.get("events", [])
                if events:
                    st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
                else:
                    st.info("Aucun evenement d'intervention n'est consigne pour cet equipement. Les brouillons apparaissent dans la section ci-dessus.")
