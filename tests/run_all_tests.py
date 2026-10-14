"""
Comprehensive Test Runner & Automated Verification Report
Runs all unit and smoke tests for Research Companion (IBM watsonx Orchestrate).
"""

import sys
import os
import unittest
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from workflows.research_companion_workflow import research_companion_workflow
from knowledge_base.kb_manager import global_kb
from tools.generate_cited_answer import STRICT_REFUSAL_MESSAGE

class TestWorkflowSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_kb.clear()
        cls.sample_files = [
            os.path.join(BASE_DIR, "sample_data", "attention_is_all_you_need.txt"),
            os.path.join(BASE_DIR, "sample_data", "rag_knowledge_intensive.txt"),
            os.path.join(BASE_DIR, "sample_data", "lora_low_rank_adaptation.txt"),
        ]
        cls.file_names = [
            "Attention Is All You Need",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "LoRA: Low-Rank Adaptation of Large Language Models"
        ]

    def test_scenario_1_upload_and_ask_methods(self):
        res = research_companion_workflow.execute(
            user_request="What methods are used in these papers?",
            uploaded_files=self.sample_files,
            file_names=self.file_names
        )
        self.assertEqual(res["status"], "success")
        ans = res["final_response"]
        self.assertIn("Attention Is All You Need", ans)
        self.assertIn("[", ans)
        self.assertIn("]", ans)

    def test_scenario_2_limitations(self):
        res = research_companion_workflow.execute(
            user_request="What are the main limitations identified by the authors?"
        )
        self.assertEqual(res["status"], "success")
        ans = res["final_response"]
        self.assertIn("[", ans)
        self.assertTrue(len(ans) > 50)

    def test_scenario_3_research_gaps(self):
        res = research_companion_workflow.execute(
            user_request="What research gaps exist?"
        )
        self.assertEqual(res["status"], "success")
        ans = res["final_response"]
        self.assertIn("[", ans)
        self.assertTrue(len(ans) > 50)

    def test_scenario_4_generate_literature_review(self):
        res = research_companion_workflow.execute(
            user_request="Generate a literature review."
        )
        self.assertEqual(res["status"], "success")
        ans = res["final_response"]
        self.assertIn("1. Background", ans)
        self.assertIn("2. Methods and Approaches", ans)
        self.assertIn("3. Key Findings", ans)
        self.assertIn("4. Limitations", ans)
        self.assertIn("5. Research Gaps", ans)
        self.assertIn("6. Future Research Directions", ans)

    def test_scenario_5_unrelated_question_refusal(self):
        res = research_companion_workflow.execute(
            user_request="How do I bake a chocolate chip sourdough cake at 350 degrees?"
        )
        self.assertEqual(res["status"], "success")
        ans = res["final_response"].strip()
        self.assertEqual(ans, STRICT_REFUSAL_MESSAGE)


def run_tests_and_report():
    print("=" * 70)
    print(" IBM watsonx Orchestrate Multi-Agent Solution: Research Companion")
    print(" Problem Statement 7: Research Agent - Automated Test Suite")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWorkflowSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print(" FINAL TEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run : {result.testsRun}")
    print(f"Successes       : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures        : {len(result.failures)}")
    print(f"Errors          : {len(result.errors)}")
    print(f"Execution Time  : {elapsed} seconds")
    
    if result.wasSuccessful():
        print("STATUS          : ALL TESTS PASSED (100% SUCCESS RATE)")
    else:
        print("STATUS          : SOME TESTS FAILED")
    print("=" * 70)

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests_and_report()
    sys.exit(0 if success else 1)
