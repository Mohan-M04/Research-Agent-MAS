"""Workflow validation script."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow.research_companion_workflow import ResearchCompanionWorkflow

wf = ResearchCompanionWorkflow()
defn = ResearchCompanionWorkflow.get_workflow_definition()

print("=== Workflow Definition Validation ===")
print("Name:   ", defn["name"])
print("Steps:  ", len(defn["steps"]), "(expected 6)")
print("Agents: ", defn["agents"])
print()

for step in defn["steps"]:
    print("  Step %d: %s (%s)" % (step["step"], step["name"], step["type"]))

assert defn["name"] == "research_companion_workflow", "Wrong name"
assert len(defn["steps"]) == 6, "Wrong step count"
assert len(defn["agents"]) == 4, "Wrong agent count"

print()
print("Workflow definition: VALID — all assertions passed.")
