from __future__ import annotations

import sys
import tempfile
import unittest
import os
from unittest.mock import MagicMock
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from maintidoc.ingestion.chunker import chunk_records
from maintidoc.ingestion import pdf_loader
from maintidoc.retrieval.retriever import prepare_evidence, filter_relevant_images, rerank_text_results, reciprocal_rank_fusion, retrieve_evidence
from maintidoc.agent.workflow import build_workflow, classify_question
from maintidoc.retrieval import vector_search
from maintidoc.vector_store import qdrant_store
from qdrant_client import QdrantClient
from maintidoc.mcp import server as mcp_server
from maintidoc.generation import qwen


MANUAL = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"
)


class ManualImageIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_bytes = MANUAL.read_bytes()

    def _load(self, image_dir: Path, *, render_vector_pages: bool = False):
        with patch.object(
            pdf_loader,
            "_run_ocr",
            side_effect=lambda _page, native_text: (native_text, False),
        ):
            return pdf_loader.load_pdf(
                self.pdf_bytes,
                MANUAL.name,
                image_dir=image_dir,
                render_vector_pages=render_vector_pages,
            )

    def test_default_ingestion_extracts_only_six_original_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._load(Path(temp_dir))

        self.assertEqual(result["page_count"], 9)
        self.assertEqual(len(result["images"]), 6)
        self.assertEqual(
            [image["page"] for image in result["images"]],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertTrue(
            all(image["type"] == "image" for image in result["images"])
        )
        self.assertTrue(
            all(
                image["metadata"]["resource_type"] == "original_image"
                and image["metadata"]["pdf_xref"]
                and image["metadata"]["image_sha256"]
                for image in result["images"]
            )
        )
        alarm_table = next(
            table for table in result["tables"] if table["page"] == 5
        )
        self.assertEqual(alarm_table["rows"][1][0], "AL-01")
        self.assertEqual(alarm_table["rows"][5][0], "AL-00")
        self.assertEqual(alarm_table["rows"][5][1], "Vert")

    def test_optional_vector_renders_are_separate_from_original_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._load(Path(temp_dir), render_vector_pages=True)

        originals = [image for image in result["images"] if image["type"] == "image"]
        renders = [
            image for image in result["images"]
            if image["type"] == "page_render"
        ]

        self.assertEqual(len(originals), 6)
        self.assertEqual(len(renders), 8)
        self.assertEqual(
            [image["page"] for image in renders],
            [1, 2, 3, 4, 5, 6, 7, 9],
        )


class OcrConfigurationTests(unittest.TestCase):
    def test_tessdata_path_is_read_when_ocr_runs(self):
        with tempfile.TemporaryDirectory() as tessdata_dir, patch.dict(
            os.environ,
            {"TESSDATA_PREFIX": tessdata_dir},
        ), patch.object(pdf_loader, "TESSDATA_DIR", None):
            page = MagicMock()
            page.get_textpage_ocr.return_value = object()
            page.get_text.return_value = "texte reconnu"

            text, attempted = pdf_loader._run_ocr(page, "")

        self.assertEqual(text, "texte reconnu")
        self.assertTrue(attempted)
        self.assertEqual(
            page.get_textpage_ocr.call_args.kwargs["tessdata"],
            tessdata_dir,
        )

    def test_ocr_provenance_survives_chunking(self):
        chunks = chunk_records([{
            "id": "manual-p8-text",
            "page": 8,
            "type": "text",
            "filename": "manual.pdf",
            "text": "Texte reconnu par OCR.",
            "ocr_attempted": True,
            "extraction_method": "native+ocr",
        }])

        self.assertTrue(chunks[0]["ocr_attempted"])
        self.assertEqual(chunks[0]["extraction_method"], "native+ocr")


class TableEvidenceTests(unittest.TestCase):
    def test_table_rows_and_type_survive_chunking_and_evidence_preparation(self):
        chunks = chunk_records([
            {
                "id": "manual-p3-table1",
                "page": 3,
                "type": "table",
                "filename": "manual.pdf",
                "text": "Code | Description\nAL-01 | Link failure",
            }
        ])

        self.assertEqual(chunks[0]["table_rows"], [
            ["Code", "Description"],
            ["AL-01", "Link failure"],
        ])

        evidence = prepare_evidence({
            "text_results": [{
                "chunk_id": chunks[0]["id"],
                "document_id": "doc-1",
                "page": 3,
                "chunk_type": "table",
                "content": chunks[0]["text"],
                "metadata": {"table_rows": chunks[0]["table_rows"]},
            }]
        })

        self.assertEqual(evidence[0]["type"], "table")
        self.assertEqual(evidence[0]["metadata"]["table_rows"], chunks[0]["table_rows"])

    def test_visual_evidence_keeps_the_image_id_expected_by_streamlit(self):
        evidence = prepare_evidence({
            "image_results": [{
                "image_id": "image-1",
                "document_id": "doc-1",
                "page": 2,
                "image_path": "image.png",
                "metadata": {"type": "image"},
            }]
        })

        self.assertEqual(evidence[0]["id"], "image-1")
        self.assertEqual(evidence[0]["image_id"], "image-1")


class VisualSearchTests(unittest.TestCase):
    def test_low_similarity_images_are_not_presented_as_relevant(self):
        results = filter_relevant_images([
            {"image_id": "figure-6", "similarity": 0.2814},
            {"image_id": "figure-5", "similarity": 0.2685},
            {"image_id": "figure-4", "similarity": 0.2575},
        ])

        self.assertEqual(
            [result["image_id"] for result in results],
            ["figure-6", "figure-5"],
        )

    def test_text_evidence_page_wins_over_slightly_higher_off_page_clip(self):
        results = filter_relevant_images([
            {"image_id": "fig-04", "page": 5, "similarity": 0.2911},
            {"image_id": "fig-01", "page": 2, "similarity": 0.2859},
        ], supporting_pages={2}, query="architecture logique SAN")
        self.assertEqual([row["image_id"] for row in results], ["fig-01"])

    def test_exact_alarm_code_promotes_matching_table_evidence(self):
        results = rerank_text_results([
            {"chunk_id": "generic", "content": "Description générale des alarmes", "similarity": 0.8},
            {"chunk_id": "alarm", "content": "AL-01 Orange Chemin FC dégradé", "similarity": 0.6},
        ], "Que signifie l'alarme AL-01 ?")
        self.assertEqual(results[0]["chunk_id"], "alarm")

    def test_reciprocal_rank_fusion_promotes_consensus(self):
        fused = reciprocal_rank_fusion([
            [{"chunk_id": "consensus"}, {"chunk_id": "vector-only"}],
            [{"chunk_id": "consensus"}, {"chunk_id": "lexical-only"}],
        ])
        self.assertEqual(fused[0]["chunk_id"], "consensus")
        self.assertEqual(fused[0]["retrieval_channels"], ["text_vector", "text_lexical"])

    def test_retrieve_evidence_fuses_text_lexical_and_visual_channels(self):
        vector_rows = [{"chunk_id": "vector", "page": 1, "similarity": 0.8}]
        lexical_rows = [{"chunk_id": "ticket-table", "page": 7, "similarity": 0, "lexical_score": 2}]
        visual_rows = [{"image_id": "figure", "page": 7, "similarity": 0.3}]
        with patch("maintidoc.retrieval.retriever.search_text", return_value=vector_rows), patch(
            "maintidoc.retrieval.retriever.search_lexical", return_value=lexical_rows
        ), patch("maintidoc.retrieval.retriever.search_visual", return_value=visual_rows):
            results = retrieve_evidence([0.0] * 1024, [0.0] * 512, query="tickets du manuel")
        self.assertIn("ticket-table", [item["chunk_id"] for item in results["text_results"]])
        self.assertEqual(results["image_results"][0]["image_id"], "figure")
        self.assertTrue(results["fused_results"])


class WorkflowRoutingTests(unittest.TestCase):
    def test_ticket_lookup_routes_to_mcp(self):
        self.assertEqual(
            classify_question("donne moi le ticket pour SAN-LAB-2000"),
            ("open_tickets", "get_open_tickets", "SAN-LAB-2000"),
        )

    def test_ticket_how_to_stays_documentary(self):
        self.assertEqual(
            classify_question("comment créer un ticket dans le manuel ?")[0],
            "document",
        )
        self.assertEqual(
            classify_question("Comment préparer un ticket pour SAN-LAB-2000 ?")[0],
            "document",
        )

    def test_explicit_draft_request_requires_confirmation(self):
        self.assertEqual(
            classify_question("Prepare un brouillon de ticket pour SAN-LAB-2000")[1:],
            ("create_ticket", "SAN-LAB-2000"),
        )

    def test_equipment_history_routes_to_mcp(self):
        self.assertEqual(
            classify_question("Quel est l'historique des interventions de SAN-LAB-2000 ?"),
            ("equipment_history", "get_equipment_history", "SAN-LAB-2000"),
        )

    def test_graph_executes_ticket_lookup_through_mcp_only(self):
        mcp_calls = []
        retrieval_calls = []
        graph = build_workflow(
            retrieve_fn=lambda *args, **kwargs: retrieval_calls.append(args) or {},
            prepare_fn=lambda result: ([], []),
            answer_fn=lambda *args: "unused",
            mcp_fn=lambda tool, args: mcp_calls.append((tool, args)) or {"tickets": []},
        )
        result = graph.invoke({"question": "donne moi le ticket pour SAN-LAB-2000"})
        self.assertEqual(result["status"], "mcp_answered")
        self.assertEqual(mcp_calls, [("get_open_tickets", {"equipment_id": "SAN-LAB-2000"})])
        self.assertEqual(retrieval_calls, [])

    def test_force_document_keeps_technical_question_out_of_ticket_tools(self):
        mcp_calls = []
        retrieval_calls = []
        graph = build_workflow(
            retrieve_fn=lambda *args, **kwargs: retrieval_calls.append(args) or {},
            prepare_fn=lambda result: ([], []),
            answer_fn=lambda *args: "unused",
            mcp_fn=lambda *args: mcp_calls.append(args) or {},
        )
        result = graph.invoke({
            "question": "donne moi les tickets pour SAN-LAB-2000",
            "force_document": True,
        })
        self.assertEqual(result["intent"], "document")
        self.assertEqual(len(retrieval_calls), 1)  # une recherche, sans relance coûteuse
        self.assertEqual(mcp_calls, [])

    def test_graph_never_creates_ticket_without_confirmation(self):
        mcp_calls = []
        retrieve = MagicMock(side_effect=AssertionError("Le parcours ticket ne doit pas appeler le RAG."))
        prepare = MagicMock(side_effect=AssertionError("Le parcours ticket ne doit pas charger les images."))
        answer = MagicMock(side_effect=AssertionError("Le parcours ticket ne doit pas attendre Qwen."))
        graph = build_workflow(
            retrieve_fn=retrieve,
            prepare_fn=prepare,
            answer_fn=answer,
            mcp_fn=lambda *args: mcp_calls.append(args) or {},
        )
        result = graph.invoke({
            "question": "Prepare un brouillon de ticket pour SAN-LAB-2000",
            "ticket_description": "Perte intermittente du chemin FC, observation simulée.",
            "ticket_alarm_code": "AL-01",
        })
        self.assertEqual(result["status"], "draft_confirmation_required")
        self.assertEqual(result["draft_ticket"]["equipment_id"], "SAN-LAB-2000")
        self.assertIn("ticket_draft_agent", result["agent_trace"])
        self.assertIn("draft_confirmation_agent", result["agent_trace"])
        self.assertIn("AL-01", result["draft_ticket"]["title"])
        self.assertIn("Perte intermittente", result["draft_ticket"]["description"])
        retrieve.assert_not_called()
        prepare.assert_not_called()
        answer.assert_not_called()
        self.assertEqual(mcp_calls, [])

    def test_document_graph_runs_three_specialist_agents_and_validates_citation(self):
        graph = build_workflow(
            retrieve_fn=lambda *args, **kwargs: {"text_results": [{"chunk_id": "c1", "page": 1}]},
            prepare_fn=lambda result: ([{"id": "c1", "page": 1, "type": "text", "content": "preuve"}], []),
            answer_fn=lambda *args: '{"status":"answered","answer":"ok","citations":[{"id":"c1","page":1}],"uncertainties":[]}',
        )
        result = graph.invoke({"question": "Que dit le document ?"})
        for name in ("intent_agent", "retrieval_agent", "response_agent"):
            self.assertIn(name, result["agent_trace"])
        self.assertTrue(result["citation_validation"]["valid"])

    def test_mcp_write_rejects_missing_human_confirmation(self):
        with self.assertRaises(PermissionError):
            mcp_server.create_ticket("SAN-LAB-2000", "Titre", "Description", human_confirmed=False)

    def test_mcp_ticket_write_is_idempotent_and_audited(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = (17, "SAN-LAB-2000", "Titre", "Description", "draft", "2026-11-20T12:00:00+00:00")
        with patch.object(mcp_server, "create_ticket_record", return_value=({"ticket_id": "17"}, True)) as create:
            mcp_server.create_ticket("SAN-LAB-2000", "Titre", "Description", human_confirmed=True)
            mcp_server.create_ticket("SAN-LAB-2000", "Titre", "Description", human_confirmed=True)

        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].args, create.call_args_list[1].args)

    def test_clip_query_is_validated_and_sent_to_image_search(self):
        expected = [{"image_id": "img-1", "page": 2, "similarity": 0.9}]
        with patch.object(
            vector_search,
            "search_images",
            return_value=expected,
        ) as search_images:
            results = vector_search.search_visual(
                [0.0] * 512,
                top_k=3,
                document_id="doc-1",
            )

        self.assertEqual(results, expected)
        self.assertEqual(search_images.call_args.kwargs["top_k"], 3)
        self.assertEqual(search_images.call_args.kwargs["document_id"], "doc-1")
        self.assertEqual(
            len(search_images.call_args.kwargs["query_embedding"]),
            512,
        )

    def test_visual_search_rejects_wrong_clip_dimension(self):
        with self.assertRaises(ValueError):
            vector_search.search_visual([0.0] * 511)

class QdrantConfigurationTests(unittest.TestCase):
    def test_example_configuration_points_to_qdrant_without_postgres(self):
        config = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
        self.assertIn("QDRANT_URL=http://127.0.0.1:6333", config)
        self.assertIn("QDRANT_TIMEOUT=140", config)
        self.assertNotIn("POSTGRES_", config)

    def test_qdrant_stores_and_retrieves_documents_and_audited_tickets(self):
        client = QdrantClient(":memory:")
        with patch.object(qdrant_store, "_client", client):
            qdrant_store._ensure_collections()
            qdrant_store.save_document("doc-test", "manual.pdf", "a" * 64, page_count=1)
            qdrant_store.save_chunks("doc-test", [{
                "id": "chunk-test", "page": 1, "chunk_index": 1,
                "type": "table", "text": "AL-01 | Chemin FC degrade",
                "table_rows": [["Code", "Description"], ["AL-01", "Chemin FC degrade"]],
            }])
            qdrant_store.save_embeddings("doc-test", ["chunk-test"], [[0.1] * 1024])
            qdrant_store.set_document_index_status("doc-test", "indexed")
            self.assertEqual(qdrant_store.list_documents()[0]["document_id"], "doc-test")
            self.assertEqual(qdrant_store.list_documents()[0]["metadata"]["index_status"], "indexed")
            self.assertEqual(qdrant_store.search_chunks([0.1] * 1024)[0]["chunk_id"], "chunk-test")
            self.assertEqual(qdrant_store.search_chunks_lexical("AL-01")[0]["chunk_id"], "chunk-test")
            first, created = qdrant_store.create_ticket_record("SAN-LAB-2000", "Titre", "Description", "key-test")
            second, created_again = qdrant_store.create_ticket_record("SAN-LAB-2000", "Titre", "Description", "key-test")
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["ticket_id"], second["ticket_id"])
            self.assertEqual(len(qdrant_store.list_records("ticket", "SAN-LAB-2000")), 1)
            self.assertEqual(len(qdrant_store.list_records("audit")), 1)
        client.close()


class LocalQwenRequestTests(unittest.TestCase):
    def test_qwen_request_disables_thinking_and_uses_bounded_completion(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"{\\"status\\":\\"answered\\"}"}}]}'
        )
        evidence = [{"id": "chunk-1", "page": 1, "type": "text", "content": "Evidence."}]
        with patch.object(qwen, "urlopen", return_value=response) as request:
            qwen.ask_qwen("Question?", evidence, [])
        payload = __import__("json").loads(request.call_args.args[0].data)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 320)


if __name__ == "__main__":
    unittest.main()
