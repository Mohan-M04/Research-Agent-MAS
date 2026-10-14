"""
tests/smoke_test_coordinator.py
─────────────────────────────────────────────────────────────────────────────
Smoke tests for the Research Companion coordinator (research_companion_workflow).

Execution modes
───────────────
1. Against synthetic PDFs (default, no files needed):
       python tests/smoke_test_coordinator.py

2. Against real arXiv PDFs in data/papers/:
       python tests/smoke_test_coordinator.py --real-papers

Failure policy
──────────────
A scenario FAILS if any of the following are true:
  • The response is empty or whitespace.
  • The response contains "internal workflow error" (case-insensitive).
  • The response contains "internal error" (case-insensitive).
  • For Scenarios 2-5: the response does not contain at least one citation
    in the format [Something, Page N] or [Something, Chunk N].
  • For Scenarios 6-7: the response does NOT contain exactly the string
    "This information is not available in the uploaded research papers."

Pre-flight checks
─────────────────
  • pymupdf (fitz)   — required for PDF extraction
  • chromadb         — required for vector store
  • PyPDF2           — fallback PDF extractor
If any are missing the script prints the exact install command and exits 1
WITHOUT running any scenarios.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Suppress noisy logs during smoke tests ───────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
NOT_AVAILABLE_MSG = "This information is not available in the uploaded research papers."
CITATION_PATTERN = re.compile(r"\[.+?,\s*(Page|Chunk|Section|p\.)\s*\d+", re.IGNORECASE)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: verify required packages are importable
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "fitz":     ("pymupdf", "pip install pymupdf"),
    "chromadb": ("chromadb", "pip install chromadb"),
    "PyPDF2":   ("PyPDF2",   "pip install PyPDF2"),
}

def _preflight_check() -> bool:
    """Check all required packages. Print install instructions and return False on failure."""
    missing = []
    for import_name, (pkg_name, install_cmd) in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append((pkg_name, install_cmd))

    if missing:
        print(f"\n{RED}{BOLD}✗ Pre-flight failed — required packages not installed:{RESET}")
        for pkg_name, install_cmd in missing:
            print(f"  {RED}Missing:{RESET} {pkg_name}")
            print(f"  {YELLOW}Install:{RESET}  {install_cmd}")
        print(f"\n{YELLOW}Install all at once:{RESET}")
        print(f"  pip install pymupdf chromadb PyPDF2")
        print(f"\nSmoke tests cannot run until these packages are available.\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Paper fixtures (used when --real-papers is NOT set)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_1_TITLE   = "Attention Is All You Need"
PAPER_1_CONTENT = """\
Attention Is All You Need

Abstract
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely.

Methods and Approaches
The Transformer uses multi-head self-attention layers combined with position-wise feed-forward networks
and positional encoding injected into the input embeddings. Encoder and decoder stacks each comprise
six identical layers. Training used the Adam optimiser with a custom learning-rate schedule.

Key Findings
The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, outperforming
all previously published ensembles. On English-to-French the model achieves 41.0 BLEU.
Training required significantly less compute than recurrent or convolutional architectures.

Limitations
The Transformer's self-attention has quadratic complexity in sequence length, making it expensive
for very long sequences. It also requires large datasets and significant compute to train from scratch.

Research Gaps
The interpretability of attention heads remains an open question. Extending Transformers to very long
sequences efficiently without quadratic complexity is unsolved. Cross-lingual transfer is limited.

Future Research Directions
Sparse attention mechanisms such as Longformer and BigBird address sequence-length limitations.
Parameter-efficient fine-tuning methods such as LoRA reduce compute costs.
Multi-modal Transformers extend the architecture beyond text.

Published 2017. Vaswani et al. Google Brain.
"""

PAPER_2_TITLE   = "BERT Pre-training of Deep Bidirectional Transformers"
PAPER_2_CONTENT = """\
BERT Pre-training of Deep Bidirectional Transformers for Language Understanding

Abstract
We introduce BERT (Bidirectional Encoder Representations from Transformers), designed to pre-train
deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.

Methods and Approaches
BERT uses masked language modeling (MLM) — randomly masking 15 percent of input tokens — and next
sentence prediction (NSP) as pre-training objectives. Fine-tuning adds a single output layer on top
of the pre-trained representations. The model uses WordPiece tokenisation and segment embeddings.

Key Findings
BERT advances the state of the art on eleven NLP benchmarks including SQuAD question answering,
MNLI textual entailment, and named entity recognition. BERT-Large achieves 93.2 F1 on SQuAD v1.1.

Limitations
BERT requires large amounts of compute for pre-training. The NSP objective was later questioned.
BERT is autoencoding-only and cannot generate text directly. Tokenisation artefacts affect rare words.

Research Gaps
Understanding what linguistic knowledge is encoded in BERT layers is not fully resolved.
Low-resource language support and domain-specific pre-training remain active research areas.

Future Research Directions
RoBERTa removed NSP and trained longer. DistilBERT provides a smaller, faster variant.
Domain-specific BERT models (BioBERT, SciBERT) improve performance on specialised corpora.

Published 2019. Devlin et al. Google AI Language.
"""

PAPER_3_TITLE   = "Deep Residual Learning for Image Recognition"
PAPER_3_CONTENT = """\
Deep Residual Learning for Image Recognition

Abstract
We present a residual learning framework to ease the training of networks substantially deeper than
those used previously. We provide evidence that residual networks are easier to optimise and gain
accuracy from considerably increased depth.

Methods and Approaches
The core idea is skip connections (shortcut connections) that bypass one or more layers. Combined
with batch normalisation, these allow training networks of 152 layers. The CIFAR-10 experiments
use 1202-layer networks. He initialisation is used for all weights.

Key Findings
ResNet won first place in ILSVRC 2015 classification with a top-5 error of 3.57 percent.
Deeper residual networks consistently outperform shallower plain networks, demonstrating that
depth genuinely helps when degradation is addressed by residual connections.

Limitations
Increased memory consumption grows with depth. Very deep networks have diminishing accuracy returns
beyond a threshold. Training time increases substantially for 1000-layer variants.

Research Gaps
Why residual connections prevent degradation is not fully theoretically understood.
Optimal depth for a given task and dataset remains an empirical question.

Future Research Directions
Wide ResNets, ResNeXt, DenseNet, and squeeze-and-excitation networks extend the residual paradigm.
Neural architecture search automates the discovery of optimal residual configurations.

Published 2016. He et al. Microsoft Research.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic PDF builder (used when --real-papers is NOT set)
# ─────────────────────────────────────────────────────────────────────────────

def _write_minimal_pdf(path: str, title: str, content: str) -> None:
    """
    Write a minimal but structurally valid PDF using PyMuPDF (fitz).
    PyMuPDF is already required by _preflight_check, so we know it is available here.
    """
    import pymupdf  # fitz alias

    doc = pymupdf.open()
    page = doc.new_page()
    # Title
    page.insert_text((50, 50), title, fontsize=14, fontname="helv")
    # Body — wrap at ~90 chars
    y = 90
    for raw_line in content.split("\n"):
        for wrapped in textwrap.wrap(raw_line, width=95) if raw_line.strip() else [""]:
            page.insert_text((50, y), wrapped, fontsize=10, fontname="helv")
            y += 14
            if y > 780:
                page = doc.new_page()
                y = 50
    doc.save(path)
    doc.close()


def _build_synthetic_pdfs() -> list[str]:
    """Create three synthetic PDFs in a temp directory. Returns their paths."""
    tmpdir = tempfile.mkdtemp(prefix="rc_smoke_")
    paths = []
    for title, content in [
        (PAPER_1_TITLE, PAPER_1_CONTENT),
        (PAPER_2_TITLE, PAPER_2_CONTENT),
        (PAPER_3_TITLE, PAPER_3_CONTENT),
    ]:
        safe = title.replace(" ", "_").replace(":", "")[:50] + ".pdf"
        p = os.path.join(tmpdir, safe)
        _write_minimal_pdf(p, title, content)
        paths.append(p)
    return paths


def _get_real_pdfs() -> list[str]:
    """Return PDFs from data/papers/. Fail loudly if directory is empty."""
    papers_dir = PROJECT_ROOT / "data" / "papers"
    if not papers_dir.exists():
        print(f"\n{RED}✗ data/papers/ does not exist.{RESET}")
        print(f"  Create it and place at least 1 PDF there, then re-run.")
        print(f"  See data/papers/INSTRUCTIONS.md for arXiv download guidance.\n")
        sys.exit(1)

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"\n{RED}✗ No PDF files found in data/papers/.{RESET}")
        print(f"  Download arXiv PDFs and place them there.")
        print(f"  See data/papers/INSTRUCTIONS.md for details.\n")
        sys.exit(1)

    print(f"  {YELLOW}ℹ Using {len(pdfs)} real PDF(s) from data/papers/:{RESET}")
    for p in pdfs:
        print(f"    {p.name}")
    return [str(p) for p in pdfs]


# ─────────────────────────────────────────────────────────────────────────────
# Assertion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_response_valid(response: str, scenario_name: str) -> tuple[bool, str]:
    """
    Run common validity checks for all scenarios.
    Returns (ok, failure_reason).
    """
    if not response or not response.strip():
        return False, "Response is empty"
    if "internal workflow error" in response.lower():
        return False, f"Response contains 'internal workflow error': {response[:120]}"
    if "internal error" in response.lower():
        return False, f"Response contains 'internal error': {response[:120]}"
    return True, ""


def _assert_has_citation(response: str) -> tuple[bool, str]:
    """
    Check that the response contains at least one citation of the form
    [Anything, Page/Chunk/Section N] or [Anything, p. N].
    """
    if CITATION_PATTERN.search(response):
        return True, ""
    return False, (
        "Response contains no citation in the format "
        "[Paper Title, Page N] or [Paper Title, Chunk N]. "
        f"Response snippet: {response[:200]}"
    )


def _assert_exact_refusal(response: str) -> tuple[bool, str]:
    """Check that the response contains the exact NOT_AVAILABLE_MSG string."""
    if NOT_AVAILABLE_MSG in response:
        return True, ""
    return False, (
        f"Response does not contain the exact refusal string.\n"
        f"  Expected (anywhere in response): {NOT_AVAILABLE_MSG!r}\n"
        f"  Got: {response[:250]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result tracker
# ─────────────────────────────────────────────────────────────────────────────

class SmokeTestResults:
    def __init__(self):
        self.total   = 0
        self.passed  = 0
        self.failed  = 0
        self.details: list[dict] = []

    def record(
        self,
        name: str,
        passed: bool,
        reason: str = "",
        response_snippet: str = "",
        service: str = "mock",
    ) -> None:
        self.total += 1
        if passed:
            self.passed += 1
            print(f"  {GREEN}✓ PASS{RESET}  {name}")
            if response_snippet:
                print(f"         {YELLOW}↳{RESET} {response_snippet[:140]} …")
        else:
            self.failed += 1
            print(f"  {RED}✗ FAIL{RESET}  {name}")
            if reason:
                print(f"         {RED}↳ {reason}{RESET}")

        self.details.append({
            "name":             name,
            "passed":           passed,
            "reason":           reason,
            "response_snippet": response_snippet[:300] if response_snippet else "",
            "service":          service,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke-test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_tests(use_real_papers: bool = False) -> SmokeTestResults:
    results = SmokeTestResults()
    mode_label = "real arXiv PDFs" if use_real_papers else "synthetic PDFs"

    # Determine service mode for reporting
    wx_live = bool(os.getenv("WATSONX_APIKEY") and os.getenv("WATSONX_PROJECT_ID"))
    service = "watsonx.ai (live)" if wx_live else "mock/dev (no credentials)"

    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  Research Companion — Smoke Tests{RESET}")
    print(f"  Mode:    {mode_label}")
    print(f"  Service: {service}")
    print(f"{BOLD}{'='*68}{RESET}\n")

    # ── Step 0: Build or locate PDFs ─────────────────────────────────────────
    print(f"{BOLD}[ Setup ] Preparing PDF sources …{RESET}")

    if use_real_papers:
        pdf_paths = _get_real_pdfs()
        results.record("Locate real PDFs in data/papers/", True, service=service)
    else:
        try:
            pdf_paths = _build_synthetic_pdfs()
            print(f"  {YELLOW}ℹ Created {len(pdf_paths)} synthetic PDFs using PyMuPDF.{RESET}")
            for p in pdf_paths:
                print(f"    {p}")
            results.record(
                f"Create {len(pdf_paths)} synthetic PDFs with PyMuPDF",
                len(pdf_paths) == 3,
                service="local",
            )
        except Exception as exc:
            results.record("Create synthetic PDFs", False, str(exc), service="local")
            return results

    # ── Import workflow ───────────────────────────────────────────────────────
    try:
        from workflow.research_companion_workflow import ResearchCompanionWorkflow
        workflow = ResearchCompanionWorkflow()
        results.record("Import ResearchCompanionWorkflow", True, service="local")
    except Exception as exc:
        results.record("Import ResearchCompanionWorkflow", False, str(exc), service="local")
        return results

    # ── Scenario 1: Ingest PDFs ───────────────────────────────────────────────
    print(f"\n{BOLD}[ Scenario 1 ] Ingest {len(pdf_paths)} PDF research paper(s){RESET}")
    try:
        r1 = workflow.execute(
            user_message="I am uploading research papers for analysis.",
            pdf_paths=pdf_paths,
        )
        ingested = r1.get("ingested_papers", [])
        wf_error  = r1.get("error") or ""

        if wf_error and not ingested:
            results.record(
                f"Scenario 1: Ingest {len(pdf_paths)} PDF(s)",
                False,
                f"Workflow error: {wf_error}",
                service=service,
            )
        elif not ingested:
            results.record(
                "Scenario 1: Ingest PDFs",
                False,
                f"Intent={r1.get('intent')}, no papers ingested. Check PDF library and chromadb.",
                service=service,
            )
        else:
            for p in ingested:
                print(f"  {YELLOW}ℹ Ingested:{RESET} {p}")
            results.record(
                f"Scenario 1: Ingested {len(ingested)}/{len(pdf_paths)} paper(s)",
                len(ingested) == len(pdf_paths),
                "" if len(ingested) == len(pdf_paths) else f"Only {len(ingested)} of {len(pdf_paths)} ingested",
                service=service,
            )
    except Exception as exc:
        results.record("Scenario 1: Ingest PDFs", False, str(exc), service=service)

    # ── Scenario 2: Methods question ─────────────────────────────────────────
    print(f"\n{BOLD}[ Scenario 2 ] 'What methods are used in these papers?'{RESET}")
    try:
        r2   = workflow.execute(user_message="What methods are used in these papers?")
        resp = r2.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 2")
        if ok:
            ok, reason = _assert_has_citation(resp)
        results.record(
            "Scenario 2: Methods question — has answer + citation",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 2: Methods question", False, str(exc), service=service)

    # ── Scenario 3: Limitations question ─────────────────────────────────────
    print(f"\n{BOLD}[ Scenario 3 ] 'What are the main limitations identified by the authors?'{RESET}")
    try:
        r3   = workflow.execute(user_message="What are the main limitations identified by the authors?")
        resp = r3.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 3")
        if ok:
            ok, reason = _assert_has_citation(resp)
        results.record(
            "Scenario 3: Limitations question — has answer + citation",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 3: Limitations question", False, str(exc), service=service)

    # ── Scenario 4: Research gaps question ───────────────────────────────────
    print(f"\n{BOLD}[ Scenario 4 ] 'What research gaps exist?'{RESET}")
    try:
        r4   = workflow.execute(user_message="What research gaps exist?")
        resp = r4.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 4")
        if ok:
            ok, reason = _assert_has_citation(resp)
        results.record(
            "Scenario 4: Research gaps question — has answer + citation",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 4: Research gaps question", False, str(exc), service=service)

    # ── Scenario 5: Literature review ────────────────────────────────────────
    print(f"\n{BOLD}[ Scenario 5 ] 'Generate a literature review.'{RESET}")
    try:
        r5   = workflow.execute(user_message="Generate a literature review of all uploaded papers.")
        resp = r5.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 5")
        if ok:
            ok, reason = _assert_has_citation(resp)
        results.record(
            "Scenario 5: Literature review — has content + citation",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 5: Literature review", False, str(exc), service=service)

    # ── Scenario 6: Unrelated question ───────────────────────────────────────
    print(f"\n{BOLD}[ Scenario 6 ] Unrelated question (pizza topping){RESET}")
    try:
        r6   = workflow.execute(user_message="What is the best pizza topping?")
        resp = r6.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 6")
        if ok:
            ok, reason = _assert_exact_refusal(resp)
        results.record(
            "Scenario 6: Unrelated question — exact refusal string",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 6: Unrelated question", False, str(exc), service=service)

    # ── Scenario 7: Off-topic history question ────────────────────────────────
    print(f"\n{BOLD}[ Scenario 7 ] Off-topic question (ancient Rome){RESET}")
    try:
        r7   = workflow.execute(user_message="Describe the history of ancient Rome.")
        resp = r7.get("response", "")
        ok, reason = _assert_response_valid(resp, "Scenario 7")
        if ok:
            ok, reason = _assert_exact_refusal(resp)
        results.record(
            "Scenario 7: Off-topic question — exact refusal string",
            ok, reason, resp, service,
        )
    except Exception as exc:
        results.record("Scenario 7: Off-topic question", False, str(exc), service=service)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results, service)
    _write_report(results, use_real_papers, service)
    return results


def _print_summary(results: SmokeTestResults, service: str) -> None:
    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  Smoke Test Summary{RESET}")
    print(f"{'='*68}")
    print(f"  Total:    {results.total}")
    print(f"  {GREEN}Passed:   {results.passed}{RESET}")
    if results.failed:
        print(f"  {RED}Failed:   {results.failed}{RESET}")
    else:
        print(f"  Failed:   0")
    print(f"\n  Service:  {service}")
    print(f"  Embeds:   {'live watsonx.ai' if 'live' in service else 'zero-vector placeholders (no credentials)'}")
    print(f"  LLM:      {'live watsonx.ai' if 'live' in service else 'dev/test mock synthesis'}")
    print(f"{BOLD}{'='*68}{RESET}\n")


def _write_report(results: SmokeTestResults, use_real_papers: bool, service: str) -> None:
    report_path = PROJECT_ROOT / "tests" / "smoke_test_report.json"
    report = {
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode":            "real_papers" if use_real_papers else "synthetic_pdfs",
        "service":         service,
        "total":           results.total,
        "passed":          results.passed,
        "failed":          results.failed,
        "scenarios":       results.details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report → {report_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Companion smoke tests")
    parser.add_argument(
        "--real-papers",
        action="store_true",
        help="Use real arXiv PDFs from data/papers/ instead of synthetic PDFs.",
    )
    args = parser.parse_args()

    # Pre-flight: verify required packages
    if not _preflight_check():
        sys.exit(1)

    results = run_smoke_tests(use_real_papers=args.real_papers)
    sys.exit(0 if results.failed == 0 else 1)
