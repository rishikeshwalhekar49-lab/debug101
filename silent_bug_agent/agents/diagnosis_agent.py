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
  :class:`~silent_bug_agent.diagnosis.models.findings.ASTFinding`,
  :class:`~silent_bug_agent.diagnosis.models.findings.SemanticFinding`, and
  :class:`~silent_bug_agent.diagnosis.models.findings.RuntimeAlert`.
* Typed Pydantic model instances of the above.
* Mixed lists containing either dicts or model instances.

Missing or empty input categories are handled gracefully — an empty list
is a valid (zero-evidence) category.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from silent_bug_agent.agents import prompts
from silent_bug_agent.diagnosis.bridge import engine_inputs, hypotheses_from_report
from silent_bug_agent.diagnosis.inference.causal_engine import CausalEngine
from silent_bug_agent.diagnosis.inference.normalizer import FindingNormalizer
from silent_bug_agent.diagnosis.knowledge.taxonomy import TAXONOMY_REGISTRY, TaxonomyEntry
from silent_bug_agent.diagnosis.models.findings import (
    ASTFinding,
    NormalizedFinding,
    RuntimeAlert,
    SemanticFinding,
)
from silent_bug_agent.diagnosis.models.hypothesis import DiagnosisResult, Hypothesis
from silent_bug_agent.findings import Report
from silent_bug_agent.schemas import AgentState, DiagnosticHypothesis
from silent_bug_agent.semantic.client import LLMClient, LLMError
from silent_bug_agent.taxonomy.loader import load_taxonomy

logger = logging.getLogger(__name__)

# Public re-exports for convenience
__all__ = ["DiagnosisAgent"]


class DiagnosisAgent:
    """AI-powered root-cause diagnosis agent for ML/DL bugs.

    The agent is stateless across calls and safe to reuse or share between
    threads (no mutable state is stored between :meth:`diagnose` calls).

    Attributes:
        _normalizer: :class:`~silent_bug_agent.diagnosis.inference.normalizer.FindingNormalizer`
            instance used for input coercion and keyword matching.
        _engine: :class:`~silent_bug_agent.diagnosis.inference.causal_engine.CausalEngine`
            instance that performs causal scoring and ranking.
    """

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self._normalizer = FindingNormalizer()
        self._engine = CausalEngine()
        self.client = client
        logger.debug("DiagnosisAgent initialised.")

    # ------------------------------------------------------------------
    # Pipeline API: shared Report in, ranked DiagnosticHypothesis out
    # ------------------------------------------------------------------

    def diagnose_report(self, report: Report, source: str = "") -> List[DiagnosticHypothesis]:
        """Rank the bugs in a detection report: causal engine first, then LLM reasoning if available."""
        if not report.findings:
            return []
        inputs = engine_inputs(report.findings)
        result = self.diagnose(ast_findings=inputs["ast"], semantic_findings=inputs["semantic"],
                               runtime_alerts=inputs["runtime"])
        scores = {h.root_cause_id: h.confidence for h in result.hypotheses}
        hypotheses = hypotheses_from_report(report, scores)
        if self.client is not None:
            hypotheses = self._reason(hypotheses, report, source)
        return hypotheses

    def _reason(self, hypotheses: List[DiagnosticHypothesis], report: Report,
                source: str) -> List[DiagnosticHypothesis]:
        """LLM phase. It may reorder and explain, but it cannot add a bug outside the taxonomy."""
        patterns = load_taxonomy()
        known = {p.id: p for p in patterns}
        catalog = "\n".join(f"- {p.id}: {p.name}" for p in patterns)
        evidence = "\n".join(f"- {f.bug_id} [{f.severity.value}] from {f.detector}, line {f.line}: {f.message[:200]}"
                             for f in report.findings[:20])
        lines = [f.line for f in report.findings if f.line]
        prompt = prompts.diagnosis_prompt(hypotheses, catalog, prompts.numbered_source(source, lines), evidence)
        try:
            reply = self.client.complete_json(prompt, prompts.DIAGNOSIS_SYSTEM, schema=prompts.DIAGNOSIS_SCHEMA,
                                               schema_name="diagnosis")
            ranked = reply["ranked"]
        except (LLMError, KeyError, TypeError) as exc:
            report.diagnostics.append(f"diagnosis: LLM reasoning unavailable, using the causal engine only ({exc})")
            return hypotheses
        by_id = {h.bug_id: h for h in hypotheses}
        ordered: List[DiagnosticHypothesis] = []
        source_lines = source.splitlines()
        for item in ranked if isinstance(ranked, list) else []:
            bug_id = item.get("bug_id") if isinstance(item, dict) else None
            line = item.get("evidence_line") if isinstance(item, dict) else None
            if bug_id not in known or any(h.bug_id == bug_id for h in ordered):
                continue
            hypothesis = by_id.get(bug_id)
            if hypothesis is None:  # a new bug needs a real line of code behind it
                if not isinstance(line, int) or not 1 <= line <= len(source_lines):
                    continue
                hypothesis = DiagnosticHypothesis(bug_id=bug_id, description=known[bug_id].name, confidence=0.0,
                                                  extra={"line": line})
            try:
                hypothesis.confidence = round(max(0.0, min(0.99, float(item.get("confidence", 0.5)))), 3)
            except (TypeError, ValueError):
                pass
            hypothesis.description = str(item.get("explanation") or hypothesis.description)[:400]
            hypothesis.extra["diagnosed_by"] = "causal_engine+llm"
            hypothesis.extra["reasoning"] = str(reply.get("reasoning", ""))[:1200]
            ordered.append(hypothesis)
        ordered += [h for h in hypotheses if all(h.bug_id != o.bug_id for o in ordered)]
        return ordered

    def run(self, state: AgentState) -> AgentState:
        """Supervisor protocol: fill `state.hypotheses` from `state.detection_report`."""
        source = ""
        if state.target_file:
            try:
                source = Path(state.target_file).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning("could not read %s: %s", state.target_file, exc)
        state.hypotheses = self.diagnose_report(state.detection_report or Report(), source)
        return state

    def __call__(self, state: AgentState) -> AgentState:
        return self.run(state)

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
                be an :class:`~silent_bug_agent.diagnosis.models.findings.ASTFinding`
                instance or a plain ``dict`` with the same fields.  May be
                ``None`` or an empty list.
            semantic_findings: Findings from semantic code review.  Each element
                may be a :class:`~silent_bug_agent.diagnosis.models.findings.SemanticFinding`
                instance or a plain ``dict``.  May be ``None`` or empty.
            runtime_alerts: Runtime invariant violation alerts.  Each element
                may be a :class:`~silent_bug_agent.diagnosis.models.findings.RuntimeAlert`
                instance or a plain ``dict``.  May be ``None`` or empty.

        Returns:
            A :class:`~silent_bug_agent.diagnosis.models.hypothesis.DiagnosisResult`
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
            The :class:`~silent_bug_agent.diagnosis.knowledge.taxonomy.TaxonomyEntry`.

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
            hypothesis: A :class:`~silent_bug_agent.diagnosis.models.hypothesis.Hypothesis`
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
