# MaintiDoc · assistant documentaire SAN

MaintiDoc extrait les PDF (texte, tableaux, figures et OCR), indexe les preuves dans **Qdrant**, puis utilise la recherche hybride RRF et **Qwen en local** pour repondre avec citations. LangGraph orchestre les agents documentaires. MCP expose l'historique et les brouillons idempotents avec confirmation humaine. L'application ne commande aucun equipement SAN.

## Stockage Qdrant

PostgreSQL et pgvector ne sont plus utilises. Qdrant conserve les documents, les chunks BGE-M3 (1024 dimensions), les images CLIP (512 dimensions), les brouillons de tickets, les evenements d'historique et les traces d'audit dans quatre collections.

## Feuille de route P0 a P4

| Etape | Fonctionnement implemente | Verification et etat |
|---|---|---|
| **P0 — Prototype documentaire** | Charge un PDF, extrait texte, tableaux, images et OCR facultatif, puis prepare des segments avec leur page et provenance. BGE-M3 encode le texte et CLIP les images. L'interface fournie est Streamlit. | Extraction du manuel de reference, images et tableaux verifies par tests; propagation de la provenance OCR verifiee. **Tests passes. Le notebook Jupyter du prototype dessine dans la feuille de route n'est pas present.** |
| **P1 — Stockage vectoriel durable** | Qdrant remplace PostgreSQL/pgvector. Quatre collections stockent les documents, chunks texte, images et donnees de suivi (historique, brouillons, audit). Le compose ne lance que Qdrant; le volume conserve ses donnees. | Test Qdrant en memoire: creation, ecriture, recherche texte/lexicale et enregistrements idempotents/audites. **Tests passes; serveur Docker non demarre pendant cette verification.** |
| **P2 — Recherche et preuves** | Recherche hybride texte vectorielle + lexicale et recherche visuelle CLIP, fusion RRF, reclassement lexical et codes d'alarme, filtrage des images et preparation des citations a partir des preuves retrouvees. | Tests de fusion, pertinence image, code exact, dimensions CLIP et preparation des preuves. **Partiel : le reclassement est lexical avec normalisation d'accents, sans reranker semantique multilingue dedie. Precision/Recall sur un corpus complet non mesures.** |
| **P3 — Orchestration agentique** | LangGraph contient six noeuds specialises. Le parcours documentaire compte trois noeuds; le parcours de brouillon en compte trois et contourne le RAG, CLIP et Qwen. | Tests du graphe, du routage, des citations et du brouillon rapide sans appel aux modeles. **Tests automatises; l'inference Qwen pour les questions documentaires reste a valider.** |
| **P4 — Outils MCP** | Serveur MCP expose l'historique, les tickets ouverts/enregistres et la creation d'un brouillon local. La creation exige une confirmation explicite, applique une cle d'idempotence et journalise l'action dans Qdrant; aucun ticket externe ni commande SAN n'est execute. | Tests de routage, refus sans confirmation, idempotence et audit. **Tests passes avec le stockage en memoire et appels simules; serveur MCP reel non lance.** |

### Verification locale et etat courant

Verification en direct le 22 novembre 2026 : Qdrant, MCP et llama.cpp repondent sur leurs endpoints locaux; le modele expose est `maintidoc-qwen`. Qdrant contient 1 manuel, 37 chunks, 6 images et 2 enregistrements de suivi. Les 37 chunks du manuel sont maintenant `embedding_ready=true`, et son statut a ete remis a `indexed`. Une recherche lexicale reelle sur `AL-01` renvoie cinq chunks.

Le parcours `intent_agent -> retrieval_agent -> response_agent` a ete execute avec ces preuves : Qwen a produit une reponse JSON avec deux citations de la page 5 et la validation des citations a reussi. Le test a pris environ deux minutes sur cette machine; le serveur local est donc fonctionnel mais lent. Le mode de raisonnement Qwen3.5 est desactive par requete, et la sortie est limitee a 320 tokens avec un timeout de 180 secondes.

MCP `get_open_tickets` retrouve le brouillon sauvegarde; `get_equipment_history` renvoie zero evenement pour SAN-LAB-2000. La page Historique affiche maintenant ces deux types de donnees dans des sections separees. L'ecriture MCP n'a pas ete testee afin de ne pas ajouter un ticket de demonstration aux donnees reelles.

Les fichiers Python compilent. Les 12 tests de routage LangGraph et le test unitaire du parametre Qwen passent. La suite complete a ete tentee, mais le processus de test s'est arrete sur `MemoryError` en lisant le PDF de 4,9 Mo; les ressources memoire etaient insuffisantes pendant l'inference locale.

## Agents LangGraph

`maintidoc/ingestion/pipeline.py` et `maintidoc/agent/workflow.py` ne font pas le meme travail : le premier transforme un PDF en index Qdrant (extraction, decoupage, embeddings et ecriture), le second route une question deja posee vers la recherche, Qwen ou les outils MCP. Les fusionner couplerait l'ajout de manuels au traitement des conversations et rendrait l'application plus difficile a maintenir.

Le graphe declare six noeuds specialises dans `maintidoc/agent/workflow.py`. Les noeuds sont des roles de traitement LangGraph, pas six modeles LLM distincts. Les deux parcours principaux ont chacun trois etapes :

1. **Intent agent** — reconnait si la demande porte sur un manuel, un brouillon, des tickets existants ou l'historique.
2. **Retrieval agent** — recherche dans Qdrant et conserve uniquement les preuves texte/image ayant une provenance exploitable (identifiant et page).
3. **Response agent** — si des preuves existent, appelle Qwen et valide dans le meme noeud la structure JSON et les couples citation/page; sans preuve, demande une precision sans appeler Qwen.
4. **Ticket draft agent** — transforme la saisie du technicien en proposition deterministe sans RAG, chargement d'images, Qwen ni ecriture.
5. **Draft confirmation agent** — verifie les champs du brouillon et le remet a l'interface; la confirmation humaine et l'enregistrement MCP restent necessaires.
6. **MCP agent** — lit les tickets et evenements d'intervention dans Qdrant via le serveur MCP.

Parcours documentaire : `intent_agent → retrieval_agent → response_agent` (3 noeuds). Parcours ticket : `intent_agent → ticket_draft_agent → draft_confirmation_agent` (3 noeuds). Consultation MCP : `intent_agent → mcp_agent` (2 noeuds). La recherche documentaire ne se relance plus automatiquement en l'absence de preuve, ce qui evite une seconde passe couteuse; l'utilisateur peut preciser sa question et relancer.

L'URL `http://127.0.0.1:8080/v1` est une base d'API, pas une page d'accueil. Une erreur `404 File Not Found` sur cette racine est normale. Pour verifier que llama.cpp a charge le modele, ouvre `http://127.0.0.1:8080/v1/models`; l'en-tete de Streamlit sonde aussi cette route et affiche l'etat de Qwen.

## Tickets : perimetre implemente

L'application presente deux pages independantes dans la navigation laterale : **Manuels et questions** pour l'indexation et le RAG documentaire, et **Tickets** pour creer et consulter les signalements. La page Tickets n'importe pas BGE-M3, CLIP ou les fonctions de recherche. Dans cette page, le technicien saisit l'equipement, un code alarme facultatif et la description du probleme. LangGraph route directement vers `ticket_draft_agent`, puis `draft_confirmation_agent`. Le brouillon indique qu'il reprend les informations declarees et qu'aucun manuel ou equipement n'a ete consulte.

Les brouillons ne passent pas par le RAG ni Qwen. Pour les questions documentaires, MaintiDoc desactive maintenant le mode de raisonnement Qwen3.5, limite la generation a 320 tokens et plafonne l'attente a 180 secondes. Sur cette machine, une requete minimale a tout de meme pris environ 25 secondes; le debit local reste lent et depend de la memoire disponible.

Le technicien relit et peut modifier le titre et la description, puis coche la confirmation. Seule la soumission de ce formulaire appelle MCP. Le serveur valide les champs et `human_confirmed=True`, applique une cle d'idempotence et ecrit le brouillon ainsi qu'une trace d'audit dans la collection Qdrant `maintidoc_records`. La page **Tickets** offre trois vues : preparer un brouillon, consulter les tickets enregistres, et historique des interventions. La vue Historique affiche maintenant dans deux sections les brouillons/tickets et les interventions; les deux listes passent par MCP sans confondre leurs statuts.

Il s'agit d'un **brouillon local persiste dans Qdrant**, pas d'un ticket cree dans ServiceNow/Jira ni d'une intervention envoyee a un equipement SAN. `external_id` reste vide, et aucune integration externe n'est configuree. Les brouillons sont des enregistrements de type `ticket`; les interventions sont des evenements de type `history`. Ainsi, un brouillon enregistre est visible dans **Tickets → Consulter les tickets**, mais n'apparait pas dans **Historique des interventions** avant qu'un evenement d'intervention distinct soit consigne. Si l'historique est vide, l'interface le precise et indique ou retrouver le brouillon. Les tests couvrent le routage, le refus sans confirmation, l'idempotence et l'audit; l'appel au serveur MCP reel doit etre valide en lancant les services.

Creer `.env` depuis `.env.example`, puis configurer :

```env
QDRANT_URL=http://127.0.0.1:6333
QDRANT_TIMEOUT=140
LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
LOCAL_LLM_MODEL=maintidoc-qwen
LOCAL_LLM_TIMEOUT=180
LOCAL_LLM_MAX_TOKENS=320
MAINTIDOC_MCP_URL=http://127.0.0.1:8001/mcp
```

Le modele retenu est `ggml-org/Qwen3.5-4B-GGUF` quantifie en `Q4_K_M` (environ 3,03 Go), telecharge depuis Hugging Face par llama.cpp. C'est une quantification 4 bits du checkpoint Qwen officiel, pas une distillation. Elle reduit le stockage et les besoins d'inference tout en conservant le modele Qwen multimodal. Le serveur llama.cpp recupere aussi le projecteur multimodal disponible pour traiter les figures.

Ce modele est un choix adapte au prototype : il sait recevoir du texte et des images, et son fichier Q4_K_M est nettement plus compact qu'un modele de plusieurs dizaines de milliards de parametres. Dans l'application, une seule image selectionnee au maximum est envoyee a Qwen par requete documentaire. Les tickets n'appellent pas Qwen. Cela ne prouve pas que ses reponses SAN sont precises : le validateur verifie la structure et l'existence des couples ID/page cites, pas la veracite semantique de chaque phrase. Aucun jeu d'evaluation MaintiDoc (questions, reponses de reference, mesures de precision) n'est encore present.

La quantification pese environ 3,03 Go sur disque. La machine de test dispose de 8 Go de RAM; llama.cpp, BGE-M3, CLIP, Qdrant et Ollama en concurrence ont laisse moins de 1 Go disponible et provoque une erreur Windows de pagination pendant l'indexation. Fermer Ollama (inutile a ce projet) et liberer de la memoire avant l'ingestion. La page du [modele GGUF](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF) liste les quantifications et recommande `llama serve`; la [documentation multimodale de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) decrit la prise en charge du projecteur image.

Pour la distillation, MaintiDoc a besoin d'un jeu de questions/reponses SAN de qualite, de reponses de reference et d'evaluations en francais. Un checkpoint de recherche distille/prune publie sur Hugging Face signale une baisse mesuree en suivi d'instructions; il n'est donc pas choisi pour remplacer le modele de reponse documentaire. On pourra distiller/fine-tuner un modele local apres constitution et evaluation de ce jeu de donnees.

Installation Windows avec WinGet (si `winget` n'est pas directement dans le `PATH`, l'appel par son chemin fonctionne aussi) :

```powershell
winget install --id ggml.llamacpp --exact
```

Si `winget` n'est pas reconnu dans PowerShell, utilise son chemin direct :

```powershell
& "$env:LOCALAPPDATA\Microsoft\WindowsApps\winget.exe" install --id ggml.llamacpp --exact
```

Apres l'installation, ferme et rouvre PowerShell pour actualiser le `PATH`. Le paquet Windows ajoute la commande `llama-server`. La commande ci-dessous force l'execution CPU et evite l'allocation Vulkan qui a echoue sur la machine de developpement; l'inference peut etre plus lente. Au premier lancement, llama.cpp telecharge les poids depuis Hugging Face; garde ce terminal ouvert pendant l'utilisation de MaintiDoc.

Demarrer le serveur Qdrant avec Docker Desktop :

```powershell
cd .\src\lab_mahir\projet
docker compose -f .\docker-compose.qdrant.yml up -d
```

Les donnees sont conservees dans le volume Docker `maintidoc_qdrant_storage`. Ne supprimez pas ce volume si vous souhaitez garder les index. Tableau de bord : `http://127.0.0.1:6333/dashboard`.

Installer les dependances dans l'environnement du depot :

```powershell
cd C:\Users\malak\Desktop\Lab_MAHIR
.\.venv\Scripts\python.exe -m pip install -r .\src\lab_mahir\projet\requirements.txt
```

## Lancement de l'application

Depuis la racine `C:\Users\malak\Desktop\Lab_MAHIR`, demarre Qdrant (Docker Desktop doit etre actif) :

```powershell
docker compose -f .\src\lab_mahir\projet\docker-compose.qdrant.yml up -d
```

Installe llama.cpp une seule fois si necessaire, puis ouvre **trois terminaux PowerShell** depuis cette meme racine :

**Terminal 1 — modele Qwen local** (la premiere execution telecharge plusieurs Go) :

```powershell
llama-server -hf ggml-org/Qwen3.5-4B-GGUF:Q4_K_M --alias maintidoc-qwen --host 127.0.0.1 --port 8080 -c 8192 --device none --no-mmproj-offload
```

**Terminal 2 — serveur MCP** :

```powershell
$env:PYTHONPATH = ".\src\lab_mahir\projet"
.\.venv\Scripts\python.exe .\src\lab_mahir\projet\maintidoc\mcp\server.py
```

**Terminal 3 — interface Streamlit** :

```powershell
$env:PYTHONPATH = ".\src\lab_mahir\projet"
.\.venv\Scripts\streamlit.exe run .\src\lab_mahir\projet\maintidoc\ui\streamlit_app.py
```

Garde les trois terminaux ouverts. Streamlit affiche l'adresse locale de l'interface. Qdrant tourne en arriere-plan via Docker. A la premiere ingestion d'un PDF, MaintiDoc cree ses collections automatiquement. Si tu migres depuis une ancienne installation PostgreSQL, reindexe les PDF dans Qdrant pour reconstruire l'index documentaire.

L'envoi des chunks et embeddings a Qdrant utilise un delai reseau configurable de 140 secondes (`QDRANT_TIMEOUT`). Une ingestion echouee est marquee `failed` dans les metadonnees du document et ne doit pas etre consideree comme interrogeable. Apres un timeout pendant l'ajout d'un manuel, relance son ingestion depuis **Ajouter un manuel**; l'identifiant du document est stable, donc la relance reprend les memes enregistrements Qdrant.

## Tests

```powershell
.\.venv\Scripts\python.exe -m compileall -q .\src\lab_mahir\projet\maintidoc
.\.venv\Scripts\python.exe -m unittest discover -s .\src\lab_mahir\projet\tests -q
```
