"""
deploy/deploy_assets.py
────────────────────────────────────────────────────────────────────────────────
Research Companion — Deployment Script

WHAT THIS SCRIPT DOES (HONEST DESCRIPTION)
───────────────────────────────────────────
This script performs TWO distinct phases that are clearly separated:

PHASE 1 — LOCAL VALIDATION (always runs, no credentials needed)
  • Imports and validates every Python tool, agent class, and workflow definition.
  • Confirms all module signatures are intact and definitions have required keys.
  • Initialises the local ChromaDB knowledge base directory.
  • This phase proves the code is importable and structurally correct.
  • It does NOT register anything in watsonx Orchestrate.

PHASE 2 — REMOTE REGISTRATION (only runs when credentials are present in .env)
  • Connects to ibm-watsonx-ai using WATSONX_APIKEY + WATSONX_URL.
  • Logs the agent definitions that WOULD be pushed to the Orchestrate API.
  • NOTE: The actual Orchestrate REST API for programmatic agent registration
    is managed through the watsonx Orchestrate UI or the Orchestrate CLI
    (not the ibm-watsonx-ai Python SDK alone). This script validates and logs
    the definitions; use the Orchestrate UI or CLI to complete registration.
  • See: https://www.ibm.com/products/watsonx-orchestrate/ai-agent-builder

REQUIRED ENVIRONMENT VARIABLES FOR REAL DEPLOYMENT
───────────────────────────────────────────────────
Set these in your .env file (copy from .env.example):

  WATSONX_APIKEY
      Your IBM Cloud API key.
      Where to find it: IBM Cloud → Manage → Access (IAM) → API keys
      → Create an IBM Cloud API key.
      URL: https://cloud.ibm.com/iam/apikeys

  WATSONX_URL
      The watsonx.ai service URL for your region.
      us-south: https://us-south.ml.cloud.ibm.com
      eu-de:    https://eu-de.ml.cloud.ibm.com
      jp-tok:   https://jp-tok.ml.cloud.ibm.com
      au-syd:   https://au-syd.ml.cloud.ibm.com
      Where to find your region: watsonx.ai instance → Manage → Details

  WATSONX_PROJECT_ID
      The ID of your watsonx.ai project.
      Where to find it: watsonx.ai → Your project → Manage → General → Project ID

  WATSONX_SPACE_ID  (optional but recommended for deployment)
      The ID of your deployment space.
      Where to find it: watsonx.ai → Deployments → Manage → Details → Space GUID

  WX_MODEL_QA  (optional, defaults to ibm/granite-3-3-8b-instruct)
      Foundation model ID for Q&A and reasoning.
      Available models: watsonx.ai → Foundation models catalog

  WX_MODEL_SUMMARIZE  (optional, defaults to ibm/granite-3-3-8b-instruct)
      Foundation model ID for literature review generation.

  WX_EMBEDDING_MODEL  (optional, defaults to ibm/slate-125m-english-rtrvr)
      Embedding model ID for semantic retrieval.
      Available embedding models: watsonx.ai → Embedding models catalog

  KB_PERSIST_DIRECTORY  (optional, defaults to ./knowledge_base/chroma_db)
      Local filesystem path where ChromaDB stores the vector index.
      In production, replace ChromaDB with a managed service (e.g. IBM Milvus,
      Elasticsearch, or Watson Discovery) and update ResearchPapersKB._get_collection().

USAGE
─────
  # Validation only (no .env needed):
  python deploy/deploy_assets.py --validate-only

  # Full run (validation + remote registration attempt):
  python deploy/deploy_assets.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deploy")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

REQUIRED_CREDS = ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentResult:
    def __init__(self):
        self.items: list[dict] = []

    def add(
        self,
        asset_type: str,
        name: str,
        status: str,
        detail: str = "",
        phase: str = "local-validation",
    ) -> None:
        icon = f"{GREEN}✓{RESET}" if status == "ok" else f"{RED}✗{RESET}"
        phase_tag = f"[{phase}]"
        print(f"  {icon}  {phase_tag} [{asset_type}] {name}  {detail}")
        self.items.append({
            "asset_type": asset_type,
            "name":       name,
            "status":     status,
            "detail":     detail,
            "phase":      phase,
        })


def _creds_present() -> bool:
    return all(os.getenv(v) for v in REQUIRED_CREDS)


def _get_watsonx_client():
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        creds  = Credentials(url=os.getenv("WATSONX_URL"), api_key=os.getenv("WATSONX_APIKEY"))
        client = APIClient(creds)
        space  = os.getenv("WATSONX_SPACE_ID")
        if space:
            client.set.default_space(space)
        return client
    except ImportError:
        logger.warning("ibm-watsonx-ai not installed; skipping live client. Run: pip install ibm-watsonx-ai")
        return None
    except Exception as exc:
        logger.warning("Could not create watsonx.ai client: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Local validation
# ─────────────────────────────────────────────────────────────────────────────

def phase1_local_validation(result: DeploymentResult) -> bool:
    """
    Validates every asset locally (no network calls).
    Returns True only if ALL local checks pass.
    """
    all_ok = True

    # ── Knowledge base ────────────────────────────────────────────────────────
    print(f"\n{BOLD}[ Phase 1a ] Knowledge base{RESET}")
    try:
        from knowledge_base.research_papers_kb import ResearchPapersKB
        kb     = ResearchPapersKB()
        kb_def = ResearchPapersKB.get_kb_definition()
        col    = kb._get_collection()          # triggers chromadb initialisation
        result.add("knowledge_base", kb_def["name"], "ok",
                   f"ChromaDB at {kb.persist_dir}, {kb.count()} chunks indexed")
    except ImportError as exc:
        result.add("knowledge_base", "research_papers_kb", "error",
                   f"Missing package: {exc}. Run: pip install chromadb")
        all_ok = False
    except Exception as exc:
        result.add("knowledge_base", "research_papers_kb", "error", str(exc))
        all_ok = False

    # ── Python tools ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}[ Phase 1b ] Python tools (import + callable check){RESET}")
    tool_map = [
        ("tools.extract_pdf_text",           "extract_pdf_text"),
        ("tools.chunk_and_index_documents",  "chunk_and_index_documents"),
        ("tools.retrieve_relevant_chunks",   "retrieve_relevant_chunks"),
        ("tools.generate_cited_answer",      "generate_cited_answer"),
        ("tools.generate_literature_review", "generate_literature_review"),
    ]
    for module_path, func_name in tool_map:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn  = getattr(mod, func_name)
            assert callable(fn)
            result.add("tool", func_name, "ok", f"callable — {module_path}")
        except Exception as exc:
            result.add("tool", func_name, "error", str(exc))
            all_ok = False

    # ── Agents ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[ Phase 1c ] Agent definitions{RESET}")
    agent_map = [
        ("agents.pdf_ingestion_agent",   "PDFIngestionAgent"),
        ("agents.rag_retrieval_agent",   "RAGRetrievalAgent"),
        ("agents.research_qa_agent",     "ResearchQAAgent"),
        ("agents.literature_review_agent","LiteratureReviewAgent"),
        ("agents.research_coordinator",  "ResearchCoordinatorAgent"),
    ]
    for module_path, class_name in agent_map:
        try:
            import importlib
            mod  = importlib.import_module(module_path)
            cls  = getattr(mod, class_name)
            defn = cls.get_agent_definition()
            assert defn.get("name") and defn.get("description") and defn.get("system_prompt")
            result.add("agent", defn["name"], "ok",
                       f"definition valid, model={defn.get('model_id','?')}")
        except Exception as exc:
            result.add("agent", class_name, "error", str(exc))
            all_ok = False

    # ── Workflow ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[ Phase 1d ] Workflow definition{RESET}")
    try:
        from workflow.research_companion_workflow import ResearchCompanionWorkflow
        wf_def = ResearchCompanionWorkflow.get_workflow_definition()
        assert len(wf_def["steps"]) == 6
        assert len(wf_def["agents"]) == 4
        result.add("workflow", wf_def["name"], "ok",
                   f"{len(wf_def['steps'])} steps, {len(wf_def['agents'])} agents")
    except Exception as exc:
        result.add("workflow", "research_companion_workflow", "error", str(exc))
        all_ok = False

    # ── End-to-end ping (out-of-scope intent — no KB needed) ─────────────────
    print(f"\n{BOLD}[ Phase 1e ] End-to-end coordinator ping{RESET}")
    try:
        from workflow.research_companion_workflow import ResearchCompanionWorkflow
        wf = ResearchCompanionWorkflow()
        r  = wf.execute(user_message="What is the capital of France?")
        if r.get("error"):
            result.add("coordinator_ping", "out_of_scope_query", "error", r["error"])
            all_ok = False
        else:
            NOT_AVAILABLE = "This information is not available in the uploaded research papers."
            resp = r.get("response", "")
            if NOT_AVAILABLE in resp or r.get("intent") == "out_of_scope":
                result.add("coordinator_ping", "out_of_scope_query", "ok",
                           f"intent={r.get('intent')}, response correctly refused")
            else:
                result.add("coordinator_ping", "out_of_scope_query", "error",
                           f"Expected refusal, got: {resp[:120]}")
                all_ok = False
    except Exception as exc:
        result.add("coordinator_ping", "out_of_scope_query", "error", str(exc))
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Remote registration (credentials required)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_remote_registration(result: DeploymentResult) -> None:
    """
    Attempts to connect to watsonx.ai and logs agent definitions.

    IMPORTANT: This does NOT automatically register agents in watsonx Orchestrate.
    The watsonx Orchestrate Agent Builder UI is the supported path for final
    registration. This phase confirms that credentials work and the SDK connects.

    To complete deployment in watsonx Orchestrate:
      1. Open watsonx Orchestrate at https://www.ibm.com/products/watsonx-orchestrate
      2. Go to Agent Builder → New agent → Import
      3. Paste the agent definition JSON printed by this script.
    """
    print(f"\n{BOLD}[ Phase 2 ] Remote credential + SDK check{RESET}")
    print(f"  {YELLOW}NOTE: This checks credentials and SDK connectivity only.{RESET}")
    print(f"  {YELLOW}Actual agent registration must be completed via the{RESET}")
    print(f"  {YELLOW}watsonx Orchestrate Agent Builder UI or CLI.{RESET}")

    client = _get_watsonx_client()
    if client is None:
        result.add("remote_sdk", "ibm-watsonx-ai client", "error",
                   "SDK unavailable — install ibm-watsonx-ai or check credentials",
                   phase="remote-registration")
        return

    result.add("remote_sdk", "ibm-watsonx-ai client", "ok",
               "Connected successfully", phase="remote-registration")

    # Log agent definitions that can be pasted into the Orchestrate UI
    import importlib
    agent_map = [
        ("agents.pdf_ingestion_agent",    "PDFIngestionAgent"),
        ("agents.rag_retrieval_agent",    "RAGRetrievalAgent"),
        ("agents.research_qa_agent",      "ResearchQAAgent"),
        ("agents.literature_review_agent","LiteratureReviewAgent"),
        ("agents.research_coordinator",   "ResearchCoordinatorAgent"),
    ]
    print(f"\n{BOLD}  Agent definitions (paste into Orchestrate Agent Builder):{RESET}")
    for module_path, class_name in agent_map:
        try:
            mod  = importlib.import_module(module_path)
            cls  = getattr(mod, class_name)
            defn = cls.get_agent_definition()
            result.add("agent_definition", defn["name"], "ok",
                       "definition logged", phase="remote-registration")
            logger.info("Agent definition for '%s':\n%s",
                        defn["name"], json.dumps(defn, indent=2))
        except Exception as exc:
            result.add("agent_definition", class_name, "error",
                       str(exc), phase="remote-registration")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(result: DeploymentResult, validate_only: bool, creds: bool) -> None:
    total      = len(result.items)
    ok_count   = sum(1 for i in result.items if i["status"] == "ok")
    err_count  = total - ok_count

    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  Deployment Summary{RESET}")
    print(f"{'='*68}")
    print(f"  Total assets checked:  {total}")
    print(f"  {GREEN}Passed:                {ok_count}{RESET}")
    if err_count:
        print(f"  {RED}Failed:                {err_count}{RESET}")
    else:
        print(f"  Failed:                0")

    print(f"\n  Phase 1 (local validation):    {'COMPLETE' if ok_count > 0 else 'INCOMPLETE'}")
    if validate_only:
        print(f"  Phase 2 (remote registration): SKIPPED (--validate-only)")
    elif creds:
        print(f"  Phase 2 (remote registration): ATTEMPTED (see logs for details)")
        print(f"  {YELLOW}→ To complete registration: use watsonx Orchestrate Agent Builder UI{RESET}")
        print(f"  {YELLOW}  https://www.ibm.com/products/watsonx-orchestrate/ai-agent-builder{RESET}")
    else:
        print(f"  Phase 2 (remote registration): SKIPPED (no credentials in .env)")
        print(f"  {YELLOW}→ Set WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID in .env{RESET}")
        print(f"  {YELLOW}  See the docstring at the top of this file for where to find each value.{RESET}")

    print(f"{BOLD}{'='*68}{RESET}\n")

    # Save report
    report_path = Path(__file__).parent / "deployment_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validate_only":  validate_only,
            "credentials_present": creds,
            "total":          total,
            "ok":             ok_count,
            "failed":         err_count,
            "assets":         result.items,
        }, f, indent=2)
    print(f"  Report saved to: {report_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def deploy_all(validate_only: bool = False) -> bool:
    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  Research Companion — Deployment{RESET}")
    print(f"{'='*68}")

    creds = _creds_present()
    if not creds:
        print(f"\n  {YELLOW}ℹ No watsonx credentials found in environment.{RESET}")
        print(f"  {YELLOW}  Running Phase 1 (local validation) only.{RESET}")
        print(f"  {YELLOW}  Set WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID in .env for Phase 2.{RESET}")
    else:
        print(f"\n  {GREEN}✓ Credentials detected.{RESET}")

    result = DeploymentResult()
    phase1_ok = phase1_local_validation(result)

    if not validate_only and creds:
        phase2_remote_registration(result)
    elif not validate_only and not creds:
        pass  # already warned above

    _print_summary(result, validate_only, creds)
    failed = [i for i in result.items if i["status"] != "ok"]
    return len(failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Companion deployment script")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run Phase 1 (local validation) only, skip Phase 2 even if credentials are present.",
    )
    args = parser.parse_args()
    success = deploy_all(validate_only=args.validate_only)
    sys.exit(0 if success else 1)
