#!/usr/bin/env python3
"""
adk/import_all.py
──────────────────
Import all Research Companion assets into the active watsonx Orchestrate
environment in the correct dependency order:

  1. Python tools  (extract_pdf_text, chunk_and_index_documents,
                    retrieve_relevant_chunks, generate_cited_answer,
                    generate_literature_review)
  2. Knowledge base (research_papers_kb)
  3. Agents        (pdf_ingestion_agent, rag_retrieval_agent,
                    research_qa_agent, literature_review_agent,
                    research_coordinator — coordinator last)

Usage (run from the research_companion/ directory):
  cd research_companion

  # Step 0 — one-time environment setup (skip if already done):
  orchestrate env add --name prod --url https://<your-wx-orchestrate-url> --activate
  # — OR — for local Developer Edition (requires Docker/Podman):
  orchestrate server start
  orchestrate env activate local

  # Step 1 — import everything:
  python adk/import_all.py

  # Dry-run (prints commands, makes no changes):
  python adk/import_all.py --dry-run

Flags:
  --dry-run      Print commands without executing them.
  --skip-tools   Skip tool import step.
  --skip-kb      Skip knowledge-base import step.
  --skip-agents  Skip agent import step.
  --env NAME     Activate a named environment before importing (optional).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Paths (always relative to research_companion/) ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent      # .../research_companion/adk/
PROJECT_ROOT = SCRIPT_DIR.parent                   # .../research_companion/

TOOL_FILES = [
    "adk/tools/extract_pdf_text_tool.py",
    "adk/tools/chunk_and_index_documents_tool.py",
    "adk/tools/retrieve_relevant_chunks_tool.py",
    "adk/tools/generate_cited_answer_tool.py",
    "adk/tools/generate_literature_review_tool.py",
]
REQUIREMENTS_FILE = "adk/tools/requirements.txt"
KB_FILE           = "adk/knowledge_bases/research_papers_kb.yaml"
AGENT_FILES = [
    "adk/agents/pdf_ingestion_agent.yaml",
    "adk/agents/rag_retrieval_agent.yaml",
    "adk/agents/research_qa_agent.yaml",
    "adk/agents/literature_review_agent.yaml",
    "adk/agents/research_coordinator.yaml",   # coordinator last (needs collaborators)
]

# ──────────────────────────────────────────────────────────────────────────────

def _check_env_active(dry_run: bool) -> bool:
    """Verify that an orchestrate environment is active. Return False if not."""
    if dry_run:
        return True
    result = subprocess.run(
        ["orchestrate", "env", "list"],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    output = result.stdout + result.stderr
    if "No active environment" in output:
        print("\n❌  No active orchestrate environment found.")
        print("    Run ONE of the following before importing:\n")
        print("    Option A — watsonx Orchestrate SaaS / on-prem:")
        print("      orchestrate env add --name prod \\")
        print("        --url https://<your-wx-orchestrate-hostname> --activate\n")
        print("    Option B — local Developer Edition (requires Docker or Podman):")
        print("      orchestrate server start")
        print("      orchestrate env activate local\n")
        return False
    return True


def _check_credentials() -> bool:
    """
    Warn if WATSONX_APIKEY / WATSONX_PROJECT_ID are absent.
    The tools will still import, but LLM calls will fall back to mock mode.
    """
    api_key    = os.environ.get("WATSONX_APIKEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    url        = os.environ.get("WATSONX_URL", "")

    # Try loading from .env if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
            api_key    = api_key    or os.environ.get("WATSONX_APIKEY", "")
            project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
            url        = url        or os.environ.get("WATSONX_URL", "")
        except ImportError:
            pass

    missing = []
    if not api_key:
        missing.append("WATSONX_APIKEY")
    if not project_id:
        missing.append("WATSONX_PROJECT_ID")
    if not url:
        missing.append("WATSONX_URL")

    if missing:
        print(f"\n⚠️   Missing credentials: {', '.join(missing)}")
        print("    Tools will import but LLM / embedding calls will use mock mode.")
        print("    Copy .env.example → .env and fill in your credentials to enable")
        print("    real watsonx.ai inference.\n")
        return False
    return True


def run(cmd: list[str], dry_run: bool, label: str = "") -> bool:
    """Execute a CLI command from PROJECT_ROOT; return True on success."""
    display = label or " ".join(cmd)
    print(f"\n  ▶  {' '.join(cmd)}")
    if dry_run:
        print("     [dry-run — not executed]")
        return True
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"     ✗  Failed (exit {result.returncode})")
        return False
    print("     ✓  OK")
    return True


def import_tools(dry_run: bool) -> bool:
    print("\n" + "═" * 60)
    print("  STEP 1 — Importing Python tools")
    print("═" * 60)
    ok = True
    for tool_file in TOOL_FILES:
        name = Path(tool_file).stem.replace("_tool", "")
        print(f"\n  ── {name}")
        if not run(
            [
                "orchestrate", "tools", "import",
                "--kind", "python",
                "--file", tool_file,
                "--requirements-file", REQUIREMENTS_FILE,
                # Keep package-root as a relative path so the ZIP uploaded to
                # Orchestrate contains the correct import paths.
                "--package-root", ".",
            ],
            dry_run,
        ):
            ok = False
    return ok


def import_knowledge_base(dry_run: bool) -> bool:
    print("\n" + "═" * 60)
    print("  STEP 2 — Importing knowledge base")
    print("═" * 60)
    return run(
        ["orchestrate", "knowledge-bases", "import", "--file", KB_FILE],
        dry_run,
    )


def import_agents(dry_run: bool) -> bool:
    print("\n" + "═" * 60)
    print("  STEP 3 — Importing agents")
    print("═" * 60)
    ok = True
    for agent_file in AGENT_FILES:
        name = Path(agent_file).stem
        print(f"\n  ── {name}")
        if not run(
            ["orchestrate", "agents", "import", "--file", agent_file],
            dry_run,
        ):
            ok = False
    return ok


def verify(dry_run: bool) -> None:
    print("\n" + "═" * 60)
    print("  STEP 4 — Verification")
    print("═" * 60)
    run(["orchestrate", "tools",           "list"], dry_run)
    run(["orchestrate", "knowledge-bases", "list"], dry_run)
    run(["orchestrate", "agents",          "list"], dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Research Companion assets into watsonx Orchestrate."
    )
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--skip-tools",  action="store_true",
                        help="Skip Python tool import.")
    parser.add_argument("--skip-kb",     action="store_true",
                        help="Skip knowledge-base import.")
    parser.add_argument("--skip-agents", action="store_true",
                        help="Skip agent import.")
    parser.add_argument("--env",         metavar="NAME",
                        help="Activate this named environment before importing.")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Research Companion — ADK Asset Import               ║")
    print("╚══════════════════════════════════════════════════════╝")
    if args.dry_run:
        print("\n  [DRY-RUN MODE — no changes will be made]\n")

    # Optional: activate a named env
    if args.env:
        print(f"\n  Activating environment: {args.env}")
        run(["orchestrate", "env", "activate", args.env], args.dry_run)

    # Guard: fail fast if no env is active
    if not _check_env_active(args.dry_run):
        return 1

    # Warn (but don't block) if watsonx.ai credentials are missing
    _check_credentials()

    success = True

    if not args.skip_tools:
        if not import_tools(args.dry_run):
            success = False

    if not args.skip_kb:
        if not import_knowledge_base(args.dry_run):
            success = False

    if not args.skip_agents:
        if not import_agents(args.dry_run):
            success = False

    verify(args.dry_run)

    print("\n" + "═" * 60)
    if success:
        print("  ✅  All assets imported successfully.")
        print()
        print("  Next steps — test the coordinator:")
        print("    orchestrate chat start --agent research_coordinator")
        print()
        print("  Smoke-test questions to try:")
        print("    1. Upload 3 PDF paths and ask me to process them.")
        print("    2. What methods are used in these papers?")
        print("    3. What are the main limitations identified by the authors?")
        print("    4. What research gaps exist?")
        print("    5. Generate a literature review.")
    else:
        print("  ❌  One or more imports failed. Review errors above.")
        print("  Tip: re-run with --skip-tools / --skip-kb / --skip-agents")
        print("       to retry only the failed step.")
    print("═" * 60 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
