"""The central orchestrator: a LangGraph StateGraph over the whole pipeline.

    perception -> diagnosis -> repair -> validation -> memory -> END
                                  ^            |
                                  +- feedback -+   (rejected, budget left)

Each node has a deterministic phase and, when a key is set, an LLM phase. A run
ends in one of three honest outcomes: no bug needing a repair, a validated patch,
or a failure with the reason.
"""

from __future__ import annotations

import ast
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from ..diagnosis.bridge import category_of
from ..findings import Report, merge_reports
from ..learning.experience_store import ExperienceStore
from ..repair.loop import AutonomousRepairLoop
from ..repair.patch_utils import MalformedDiffError, apply_patch_to_file, apply_unified_diff
from ..repair.pending import bundled_example, copy_path, save_pending
from ..schemas import AgentState, AgentStatus, PatchCandidate, ValidationVerdict
from ..semantic.analyzer import SemanticAnalyzer
from ..semantic.client import LLMError
from ..semantic.router import ModelRouter
from ..static.analyzer import StaticAnalyzer
from ..validation.analyzer import ValidationAnalyzerAgent
from ..validation.multiseedagent import MultiseedAgent
from ..validation.overfit_agent import OverfitOracleAgent
from . import prompts
from .diagnosis_agent import DiagnosisAgent
from .fix_synthesizer_agent import FixSynthesizerAgent
from .meta_evolution import MetaEvolutionAgent
from .runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent

# Reported, but a repair would not change what the model learns.
REPORT_ONLY = {"unset_random_seed"}
REWARD_ACCEPTED, REWARD_REJECTED, REWARD_INVALID = 1.0, -0.5, -1.0
USER_DECLINED = ("The user was shown this patch, turned it down and asked for a different fix. "
                 "Propose a genuinely different change, not a rewording of this one.")


@dataclass
class PipelineConfig:
    max_iterations: int = 3
    monitor_steps: int = 150
    validation_steps: int = 60
    overfit_steps: int = 80
    seeds: Sequence[int] = (42, 1337, 2024, 7, 99)
    timeout_per_run: float = 180.0
    script_args: List[str] = field(default_factory=list)
    dry_run: bool = False
    keep_pending: bool = False  # with dry_run: keep the validated patch so `apply` can write it after approval
    # patches the user turned down, as {"diff", "bug_id"}: the repair must come up with something else
    rejected_patches: List[Dict[str, str]] = field(default_factory=list)
    offline: bool = False
    meta_every: int = 5
    use_vectors: bool = True


class GraphState(TypedDict):
    state: AgentState


class SupervisorAgent:
    """Runs one training script through detection, diagnosis, repair, validation and memory."""

    def __init__(self, config: Optional[PipelineConfig] = None, *, router: Optional[ModelRouter] = None,
                 memory: Optional[ExperienceStore] = None, detection_agent: Any = None,
                 emitter: Any = None) -> None:
        self.config = config or PipelineConfig()
        self.router = router or ModelRouter(offline=self.config.offline)
        self.memory = memory or ExperienceStore(use_vectors=self.config.use_vectors)
        self.emitter = emitter
        self.detection_agent = detection_agent or RuntimeMonitorAgent(
            MonitorConfig(max_steps=self.config.monitor_steps, max_seconds=self.config.timeout_per_run * 2))
        self.static = StaticAnalyzer()
        self.diagnosis_agent = DiagnosisAgent(client=self.router.task("diagnosis"))
        self.fix_agent = FixSynthesizerAgent(client=self.router.task("fix"), memory=self.memory)
        self.repair_loop = AutonomousRepairLoop(self.fix_agent)
        self.overfit = OverfitOracleAgent(steps=self.config.overfit_steps, timeout=self.config.timeout_per_run,
                                          script_args=self.config.script_args)
        self.multiseed = MultiseedAgent(seeds=list(self.config.seeds), timeout_per_run=self.config.timeout_per_run,
                                        measured=True, max_steps=self.config.validation_steps,
                                        script_args=self.config.script_args)
        self.analyzer = ValidationAnalyzerAgent(client=self.router.task("validation"))
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(GraphState)
        for name, node in (("perception", self._perception), ("diagnosis", self._diagnosis), ("repair", self._repair),
                           ("validation", self._validation), ("feedback", self._feedback), ("memory", self._memory)):
            graph.add_node(name, node)
        graph.add_edge(START, "perception")
        graph.add_conditional_edges("perception", self._after_perception, {"diagnosis": "diagnosis", "end": END})
        graph.add_conditional_edges("diagnosis", self._after_diagnosis, {"repair": "repair", "end": END})
        graph.add_conditional_edges("repair", self._after_repair,
                                    {"validation": "validation", "repair": "repair", "memory": "memory"})
        graph.add_conditional_edges("validation", self._after_validation, {"memory": "memory", "feedback": "feedback"})
        graph.add_edge("feedback", "repair")
        graph.add_edge("memory", END)
        return graph.compile()

    # ----------------------------------------------------------------- entry
    def run(self, target_file: str) -> AgentState:
        state = AgentState(run_id=uuid.uuid4().hex[:12], target_file=str(target_file),
                           max_iterations=self.config.max_iterations, status=AgentStatus.RUNNING,
                           extra={"script_args": list(self.config.script_args)})
        self._exclude_declined(state)
        result = self.graph.invoke({"state": state}, config={"recursion_limit": 20 + 8 * self.config.max_iterations})
        final: AgentState = result["state"]
        final.extra["llm"] = self.router.summary()
        final.extra["memory_notes"] = list(self.memory.notes)
        return final

    __call__ = run

    def _exclude_declined(self, state: AgentState) -> None:
        """Mark the patches the user turned down as already tried, so the same result is never offered
        again, and pass the reason to the fix agent as feedback."""
        if not self.config.rejected_patches:
            return
        try:
            original = Path(state.target_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return
        for item in self.config.rejected_patches:
            try:
                declined = apply_unified_diff(original, item["diff"])
            except (MalformedDiffError, KeyError):
                continue  # it was made for another version of the file
            state.extra.setdefault("tried_patches", []).append(self.fix_agent._digest(declined))
            state.extra.setdefault("repair_feedback", []).append(
                {"bug_id": item.get("bug_id", ""), "diff": item["diff"], "reason": USER_DECLINED})

    def run_until_clean(self, target_file: str, *, max_rounds: int = 3) -> List[AgentState]:
        """Fix one bug per round, re-detecting each time, because scripts often have several."""
        states: List[AgentState] = []
        for _ in range(max_rounds):
            state = self.run(target_file)
            states.append(state)
            if not state.extra.get("patch_applied"):
                break
        return states

    # ---------------------------------------------------------------- nodes
    def _perception(self, graph_state: GraphState) -> GraphState:
        if self.emitter:
            self.emitter.phase("perception", "start")
        state = graph_state["state"]
        path = Path(state.target_file)
        state.detection_report = self.static.analyze_file(path)
        client = self.router.task("semantic_review")
        if client is not None:
            semantic = SemanticAnalyzer(client).analyze_file(path)
            state.detection_report = merge_reports(state.detection_report, semantic)
        self.detection_agent.config.cwd = str(path.resolve().parent)
        state = self.detection_agent(state)
        self._consolidate(state)
        summary = state.extra.get("runtime_monitor") or {}
        if summary.get("script_status") == "error":
            state.status = AgentStatus.FAILED
            state.extra["outcome"] = "script_error"
            state.abort_reason = "The script itself raises an error, so it cannot be validated: " \
                                 + str(summary.get("script_error"))[-600:]
        elif not [f for f in state.detection_report.findings if f.bug_id not in REPORT_ONLY]:
            state.status = AgentStatus.DONE
            state.extra["outcome"] = "no_bug_found"
        if self.emitter:
            self.emitter.phase("perception", "done", findings=len(state.detection_report.findings))
        return {"state": state}

    def _consolidate(self, state: AgentState) -> None:
        client = self.router.task("perception")
        findings = state.detection_report.findings
        if client is None or not findings:
            return
        try:
            reply = client.complete_json(prompts.consolidate_prompt(findings[:25]), prompts.CONSOLIDATE_SYSTEM,
                                         schema=prompts.CONSOLIDATE_SCHEMA, schema_name="consolidation")
            state.extra["perception_summary"] = str(reply.get("summary", ""))[:600]
            state.extra["side_effects"] = [str(b) for b in reply.get("side_effect_bug_ids", [])]
        except (LLMError, AttributeError) as exc:
            state.detection_report.diagnostics.append(f"perception: LLM summary unavailable ({exc})")

    def _diagnosis(self, graph_state: GraphState) -> GraphState:
        if self.emitter:
            self.emitter.phase("diagnosis", "start")
        state = self.diagnosis_agent(graph_state["state"])
        repairable = [h for h in state.hypotheses if h.bug_id not in REPORT_ONLY]
        side_effects = set(state.extra.get("side_effects", []))
        primary = [h for h in repairable if h.bug_id not in side_effects]
        state.hypotheses = primary + [h for h in repairable if h.bug_id in side_effects]
        if not state.hypotheses:
            state.status = AgentStatus.FAILED
            state.abort_reason = "Diagnosis produced no repairable hypothesis."
        if self.emitter:
            self.emitter.phase("diagnosis", "done", hypotheses=len(state.hypotheses))
            for i, h in enumerate(state.hypotheses):
                self.emitter.hypothesis(
                    rank=i + 1, bug_id=h.bug_id, confidence=h.confidence,
                    description=h.description, diagnosed_by=h.extra.get("diagnosed_by")
                )
        return {"state": state}

    def _repair(self, graph_state: GraphState) -> GraphState:
        if self.emitter:
            self.emitter.phase("repair", "start")
        state = graph_state["state"]
        top = self.fix_agent._next_hypothesis(state)
        if top is not None:
            state.extra["repair_bandit"] = self.memory.load_bandit(category_of(top.bug_id))
        before = len(state.patch_candidates)
        state = self.repair_loop.run(state)
        made = len(state.patch_candidates) > before
        state.extra["candidate_ready"] = made and bool(state.extra.get("repair_loop_result", {}).get("structurally_valid"))
        if made and not state.extra["candidate_ready"]:
            candidate = state.patch_candidates[-1]
            state.extra.setdefault("repair_feedback", []).append(
                {"bug_id": candidate.hypothesis_bug_id, "diff": candidate.diff,
                 "reason": str(state.extra.get("repair_loop_result", {}).get("message"))})
            self._record(state, candidate, None)
        if self.emitter:
            if made:  # a candidate still to be validated, not an accepted patch
                cand = state.patch_candidates[-1]
                self.emitter.patch(
                    bug_id=cand.hypothesis_bug_id, strategy=cand.strategy.value,
                    source=cand.source, diff=cand.diff, confidence=cand.confidence
                )
            self.emitter.phase("repair", "done", candidate=made)
        return {"state": state}

    def _bug_gone(self, state: AgentState, candidate: PatchCandidate, patched: str) -> Optional[bool]:
        """Is the targeted bug absent when the detectors run on the patched code?"""
        bug = candidate.hypothesis_bug_id
        original = Path(state.target_file)
        if any(f.bug_id == bug for f in self.static.analyze_code(patched, str(original)).findings):
            return False
        folder = tempfile.mkdtemp(prefix="sba_check_")
        try:
            copy = Path(folder) / original.name
            copy.write_text(patched, encoding="utf-8")
            agent = RuntimeMonitorAgent(MonitorConfig(max_steps=self.config.monitor_steps,
                                                      max_seconds=self.config.timeout_per_run,
                                                      cwd=str(original.resolve().parent)))
            report = agent.monitor_script(copy, list(self.config.script_args))
        finally:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)
        if (agent.last_summary or {}).get("script_status") == "error":
            return None
        return not any(f.bug_id == bug for f in report.findings)

    def _validation(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        candidate = state.patch_candidates[-1]
        source = Path(state.target_file).read_text(encoding="utf-8")
        patched = apply_unified_diff(source, candidate.diff)
        verdicts: List[ValidationVerdict] = state.validation_verdicts
        if self.emitter:
            self.emitter.phase("gate1", "start")
        self.overfit(state)
        overfit = verdicts[-1]
        if self.emitter:
            self.emitter.phase("gate1", "done", accepted=overfit.accepted)
            if overfit.accepted:
                self.emitter.phase("gate2", "start")
        self.multiseed(state) if overfit.accepted else self._skip_gate_two(state, candidate)
        multiseed = verdicts[-1]
        if self.emitter:
            self.emitter.phase("gate2", "done", accepted=multiseed.accepted, skipped=not overfit.accepted)
        bug_gone = self._bug_gone(state, candidate, patched) if overfit.accepted else None
        hypothesis = next((h for h in state.hypotheses if h.bug_id == candidate.hypothesis_bug_id), None)
        strong = hypothesis is None or hypothesis.extra.get("max_severity", "warning") != "info"
        final = self.analyzer.analyze(candidate, overfit, multiseed, bug_gone, strong)
        verdicts.append(final)
        if self.emitter:
            self.emitter.verdict(
                bug_id=candidate.hypothesis_bug_id,
                accepted=final.accepted,
                validator=final.validator,
                p_value=multiseed.p_value,
                effect_size=multiseed.effect_size,
                notes=final.notes,
            )
        candidate.extra.update({"validated": True, "status": "accepted" if final.accepted else "rejected"})
        return {"state": state}

    @staticmethod
    def _skip_gate_two(state: AgentState, candidate: PatchCandidate) -> None:
        state.validation_verdicts.append(ValidationVerdict(
            patch_candidate=candidate, accepted=False, validator="multiseed_stats", p_value=1.0, seeds_run=0,
            notes="Skipped: Overfit Oracle rejected this candidate prior to multi-seed verification."))

    def _feedback(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        candidate = state.patch_candidates[-1]
        verdict = state.validation_verdicts[-1]
        state.extra.setdefault("repair_feedback", []).append(
            {"bug_id": candidate.hypothesis_bug_id, "diff": candidate.diff, "reason": verdict.notes})
        self._record(state, candidate, verdict)
        return {"state": state}

    def _record(self, state: AgentState, candidate: PatchCandidate, verdict: Optional[ValidationVerdict]) -> None:
        accepted = bool(verdict and verdict.accepted)
        reward = REWARD_ACCEPTED if accepted else (REWARD_REJECTED if verdict else REWARD_INVALID)
        category = category_of(candidate.hypothesis_bug_id)
        self.memory.remember({
            "run_id": state.run_id, "bug_id": candidate.hypothesis_bug_id, "category": category,
            "strategy": candidate.strategy.value, "source": candidate.source, "accepted": accepted, "reward": reward,
            "iterations": state.iteration, "summary": str(candidate.extra.get("rationale", ""))[:300],
            "diff": candidate.diff, "notes": (verdict.notes if verdict else "")[:500],
            "p_value": verdict.p_value if verdict else None, "effect_size": verdict.effect_size if verdict else None,
            "exemplar_ids": candidate.extra.get("exemplar_ids", []),
        })
        self.memory.update_bandit(category, candidate.strategy.value, reward)
        self.memory.score_lessons("fix", accepted)

    def _memory(self, graph_state: GraphState) -> GraphState:
        if self.emitter:
            self.emitter.phase("memory", "start")
        state = graph_state["state"]
        accepted = state.accepted_patches
        if accepted:
            verdict = accepted[-1]
            candidate = verdict.patch_candidate
            self._record(state, candidate, verdict)
            state.extra["outcome"] = "fixed"
            state.status = AgentStatus.DONE
            protected = bundled_example(Path(state.target_file))
            if protected:  # never edited: `apply` writes the fixed version to a copy instead
                state.extra["copy_to"] = str(copy_path(Path(state.target_file)))
            if not self.config.dry_run and not protected:
                backup = apply_patch_to_file(state.target_file, candidate.diff)
                state.extra["patch_applied"] = True
                state.extra["backup"] = str(backup)
            elif self.config.keep_pending or (protected and not self.config.dry_run):
                save_pending(Path(state.target_file), candidate.diff, bug_id=candidate.hypothesis_bug_id,
                             run_id=state.run_id)
                state.extra["pending_patch"] = True
        else:
            state.status = AgentStatus.FAILED
            state.extra.setdefault("outcome", "not_fixed")
            finals = [v for v in state.validation_verdicts if v.validator == "validation_analyzer"]
            state.abort_reason = state.abort_reason or (f"No patch passed validation. Last verdict: {finals[-1].notes}"
                                                         if finals else state.extra.get("repair_error")) \
                or "No patch passed validation."
            if state.patch_candidates and state.validation_verdicts:
                last = state.patch_candidates[-1]
                final = state.validation_verdicts[-1]
                if final.patch_candidate is last and final.validator == "validation_analyzer":
                    self._record(state, last, final)
        if self.memory.count() and self.memory.count() % self.config.meta_every == 0:
            state.extra["meta"] = MetaEvolutionAgent(self.memory, self.router.task("meta")).run().to_dict()
        if self.emitter:
            self.emitter.phase("memory", "done")
        return {"state": state}

    # --------------------------------------------------------------- routing
    @staticmethod
    def _after_perception(graph_state: GraphState) -> str:
        return "diagnosis" if graph_state["state"].status == AgentStatus.RUNNING else "end"

    @staticmethod
    def _after_diagnosis(graph_state: GraphState) -> str:
        return "repair" if graph_state["state"].status == AgentStatus.RUNNING else "end"

    def _after_repair(self, graph_state: GraphState) -> str:
        state = graph_state["state"]
        if state.extra.get("candidate_ready"):
            return "validation"
        if state.should_abort() or self.fix_agent._next_hypothesis(state) is None:
            return "memory"
        return "repair"

    @staticmethod
    def _after_validation(graph_state: GraphState) -> str:
        state = graph_state["state"]
        if state.accepted_patches or state.should_abort():
            return "memory"
        return "feedback"


def describe_state(state: AgentState) -> Dict[str, Any]:
    """A plain summary of a finished run, used by the CLI and by reports."""
    verdicts = [v for v in state.validation_verdicts if v.validator == "validation_analyzer"]
    accepted = state.accepted_patches
    return {
        "run_id": state.run_id,
        "status": state.status.value,
        "outcome": state.extra.get("outcome") or ("failed" if state.status == AgentStatus.FAILED else state.status.value),
        "reason": state.abort_reason,
        "iterations": state.iteration,
        "patch_applied": bool(state.extra.get("patch_applied")),
        "backup": state.extra.get("backup"),
        "hypotheses": [{"bug_id": h.bug_id, "confidence": h.confidence, "explanation": h.description,
                        "diagnosed_by": h.extra.get("diagnosed_by")} for h in state.hypotheses],
        "attempts": [{"bug_id": v.patch_candidate.hypothesis_bug_id, "source": v.patch_candidate.source, "accepted": v.accepted,
                      "validator": v.validator, "route": v.extra.get("route"), "p_value": v.p_value,
                      "effect_size": v.effect_size, "notes": v.notes}
                     for v in verdicts],
        "accepted_diff": accepted[-1].patch_candidate.diff if accepted else None,
        "accepted_patch": ({"bug_id": accepted[-1].patch_candidate.hypothesis_bug_id,
                            "strategy": accepted[-1].patch_candidate.strategy.value,
                            "source": accepted[-1].patch_candidate.source,
                            "confidence": accepted[-1].patch_candidate.confidence} if accepted else None),
        "pending_patch": bool(state.extra.get("pending_patch")),
        "copy_to": state.extra.get("copy_to"),
        "llm": state.extra.get("llm"),
        "llm_notes": state.extra.get("llm_notes", []),
        "summary": state.extra.get("perception_summary"),
        "meta": state.extra.get("meta"),
        "memory_notes": state.extra.get("memory_notes", []),
    }
