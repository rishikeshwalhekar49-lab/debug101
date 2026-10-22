
import json

from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent

# Create the diagnosis agent
agent = DiagnosisAgent()

# Provide sample findings from two analysis sources
result = agent.diagnose(
    ast_findings=[
        {
            "id": "AST-001",
            "type": "gradient_management_error",
            "message": "optimizer.zero_grad() may be missing",
            "file": "train.py",
            "line": 42,
            "severity": "high",
        }
    ],
    semantic_findings=[],
    runtime_alerts=[
        {
            "id": "RUN-001",
            "type": "gradient_instability",
            "message": "Gradient norm increased sharply",
            "metadata": {"gradient_norm": 18000},
        }
    ],
)

# Display the complete diagnosis as readable JSON
print("===== SILENT BUG AGENT: DIAGNOSIS REPORT =====")
print(result.to_json(indent=2))

# Display the leading hypothesis
top = result.primary_hypothesis

if top:
    print("\n===== PRIMARY ROOT-CAUSE HYPOTHESIS =====")
    print(f"Root cause: {top.root_cause}")
    print(f"Confidence score: {top.confidence:.2f}")
    print(f"Explanation: {top.explanation}")

    print("\nVerification steps:")
    for check in top.verification_checks:
        print(f"- {check}")

    print("\nCausal chain:")
    print(agent.explain_causal_chain(top))
else:
    print("\nNo primary hypothesis was produced.")
