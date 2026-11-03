"""
Root-Cause Diagnosis Agent for ML/DL bugs.

This is the public entry point for the Silent Bug Agent diagnosis pipeline.
It accepts three categories of findings, runs them through normalisation and
causal inference, and returns ranked, explainable hypotheses.

Typical usage::

    from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent()

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

    print(result.to_json())

The agent accepts:

* Plain ``dict`` objects with the fields documented in
  :class:`~silent_bug_agent.models.findings.ASTFinding`,
  :class:`~silent_bug_agent.models.findings.SemanticFinding`, and
  :class:`~silent_bug_agent.models.findings.RuntimeAlert`.
* Typed Pydantic model instances of the above.
* Mixed lists containing either dicts or model instances.

Missing or empty input categories are handled gracefully — an empty list
is a valid (zero-evidence) category.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from silent_bug_agent.inference.causal_engine import CausalEngine
from silent_bug_agent.inference.normalizer import FindingNormalizer
from silent_bug_agent.knowledge.taxonomy import TAXONOMY_REGISTRY, TaxonomyEntry
from silent_bug_agent.models.findings import (
    ASTFinding,
    NormalizedFinding,
    RuntimeAlert,
    SemanticFinding,
)
from silent_bug_agent.models.hypothesis import DiagnosisResult, Hypothesis

logger = logging.getLogger(__name__)

# Public re-exports for convenience
__all__ = ["DiagnosisAgent"]


class DiagnosisAgent:
    """AI-powered root-cause diagnosis agent for ML/DL bugs.

    The agent is stateless across calls and safe to reuse or share between
    threads (no mutable state is stored between :meth:`diagnose` calls).

    Attributes:
        _normalizer: :class:`~silent_bug_agent.inference.normalizer.FindingNormalizer`
            instance used for input coercion and keyword matching.
        _engine: :class:`~silent_bug_agent.inference.causal_engine.CausalEngine`
            instance that performs causal scoring and ranking.
    """

    def __init__(self) -> None:
        self._normalizer = FindingNormalizer()
        self._engine = CausalEngine()
        logger.debug("DiagnosisAgent initialised.")

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        ast_findings: Optional[Sequence[Union[ASTFinding, Dict[str, Any]]]] = None,
        semantic_findings: Optional[Sequence[Union[SemanticFinding, Dict[str, Any]]]] = None,
        runtime_alerts: Optional[Sequence[Union[RuntimeAlert, Dict[str, Any]]]] = None,
    ) -> DiagnosisResult:
        """Diagnose the root cause(s) of ML/DL bugs from multi-source findings.

        Args:
            ast_findings: Findings from static AST analysis.  Each element may
                be an :class:`~silent_bug_agent.models.findings.ASTFinding`
                instance or a plain ``dict`` with the same fields.  May be
                ``None`` or an empty list.
            semantic_findings: Findings from semantic code review.  Each element
                may be a :class:`~silent_bug_agent.models.findings.SemanticFinding`
                instance or a plain ``dict``.  May be ``None`` or empty.
            runtime_alerts: Runtime invariant violation alerts.  Each element
                may be a :class:`~silent_bug_agent.models.findings.RuntimeAlert`
                instance or a plain ``dict``.  May be ``None`` or empty.

        Returns:
            A :class:`~silent_bug_agent.models.hypothesis.DiagnosisResult`
            containing ranked hypotheses, diagnosis status, evidence summary,
            and any warnings.

        Note:
            This method never raises on malformed individual findings — invalid
            items are logged and skipped.  An exception is only raised for
            catastrophic internal errors (bugs in this code) that should not be
            silently swallowed.
        """
        ast_raw = list(ast_findings) if ast_findings else []
        sem_raw = list(semantic_findings) if semantic_findings else []
        run_raw = list(runtime_alerts) if runtime_alerts else []

        logger.info(
            "DiagnosisAgent.diagnose: %d AST, %d semantic, %d runtime findings.",
            len(ast_raw),
            len(sem_raw),
            len(run_raw),
        )

        # --- Normalise ---
        norm_ast: List[NormalizedFinding] = self._normalizer.normalize_ast(ast_raw)
        norm_sem: List[NormalizedFinding] = self._normalizer.normalize_semantic(sem_raw)
        norm_run: List[NormalizedFinding] = self._normalizer.normalize_runtime(run_raw)

        logger.debug(
            "Normalised: %d AST, %d semantic, %d runtime.",
            len(norm_ast),
            len(norm_sem),
            len(norm_run),
        )

        # --- Causal inference ---
        result: DiagnosisResult = self._engine.infer(norm_ast, norm_sem, norm_run)

        logger.info(
            "Diagnosis complete: status=%s, top_hypothesis=%s, confidence=%s.",
            result.status.value,
            result.primary_hypothesis.root_cause_id if result.primary_hypothesis else "none",
            result.primary_hypothesis.confidence if result.primary_hypothesis else "n/a",
        )

        return result

    # ------------------------------------------------------------------
    # Taxonomy access
    # ------------------------------------------------------------------

    def lookup_taxonomy(self, category_id: str) -> TaxonomyEntry:
        """Return the taxonomy entry for a given root-cause category ID.

        Args:
            category_id: A stable taxonomy identifier such as
                ``"gradient_management_error"`` or ``"numerical_instability"``.

        Returns:
            The :class:`~silent_bug_agent.knowledge.taxonomy.TaxonomyEntry`.

        Raises:
            KeyError: If *category_id* is not in the taxonomy.
        """
        return TAXONOMY_REGISTRY.lookup(category_id)

    def list_taxonomy_categories(self) -> List[str]:
        """Return all valid taxonomy category IDs in sorted order."""
        return sorted(TAXONOMY_REGISTRY.category_ids())

    # ------------------------------------------------------------------
    # Causal explanation helper
    # ------------------------------------------------------------------

    def explain_causal_chain(self, hypothesis: Hypothesis) -> str:
        """Return a formatted narrative of the causal chain in a hypothesis.

        Args:
            hypothesis: A :class:`~silent_bug_agent.models.hypothesis.Hypothesis`
                from a :meth:`diagnose` result.

        Returns:
            A multi-line string describing the causal chain.
        """
        if not hypothesis.causal_chain:
            return f"No causal chain available for {hypothesis.root_cause!r}."
        lines = [f"Causal chain for '{hypothesis.root_cause}':"]
        for i, step in enumerate(hypothesis.causal_chain, start=1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)
