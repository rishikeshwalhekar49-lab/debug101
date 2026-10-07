"""Multi-agent shared state schema and inter-agent data contracts.

This module defines the canonical data types that flow between every agent
in the pipeline:

    Detection agents  -->  Finding          (already in findings.py)
    Diagnosis agent   -->  DiagnosticHypothesis
    Fix agent         -->  PatchCandidate
    Validation agents -->  ValidationVerdict
    Supervisor        -->  AgentState  (the LangGraph node state)

All downstream agents (Rishikesh's RuntimeMonitorAgent, Person 3's
Validation agents, Person 4's DiagnosisAgent / FixSynthesizerAgent) must
import from here so the whole pipeline shares one contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .findings import Finding, Report


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    ABORTED  = "aborted"  # supervisor early-exit


class RepairStrategy(str, Enum):
    DIRECT_EDIT      = "direct_edit"       # minimal line patch
    REFACTOR         = "refactor"           # structural rewrite
    HYPERPARAM       = "hyperparam"         # learning-rate / config tweak
    FRAMEWORK_CONFIG = "framework_config"   # DDP / HF args fix
    UNKNOWN          = "unknown"


# ---------------------------------------------------------------------------
# DiagnosticHypothesis  (emitted by Root-Cause Diagnosis Agent — Person 4)
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticHypothesis:
    """A single root-cause hypothesis ranked by confidence.

    Person 4's DiagnosisAgent emits a *list* of these, sorted descending
    by confidence, for the FixSynthesizerAgent to consume.
    """
    bug_id: str                  # taxonomy entry, e.g. "BUG_MISSING_ZERO_GRAD"
    description: str             # human-readable root-cause explanation
    confidence: float            # 0..1
    evidence_finding_keys: List[tuple] = field(default_factory=list)
    # ^ (bug_id, file, line) tuples from the upstream Finding objects
    suggested_strategy: RepairStrategy = RepairStrategy.UNKNOWN
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "description": self.description,
            "confidence": self.confidence,
            "evidence_finding_keys": [list(k) for k in self.evidence_finding_keys],
            "suggested_strategy": self.suggested_strategy.value,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagnosticHypothesis":
        d = dict(d)
        d["evidence_finding_keys"] = [tuple(k) for k in d.get("evidence_finding_keys", [])]
        d["suggested_strategy"] = RepairStrategy(d.get("suggested_strategy", "unknown"))
        return cls(**d)


# ---------------------------------------------------------------------------
# PatchCandidate  (emitted by Fix Synthesizer Agent — Person 4)
# ---------------------------------------------------------------------------

@dataclass
class PatchCandidate:
    """A proposed unified-diff patch that addresses a root cause.

    The FixSynthesizerAgent may emit several candidates.  The Bandit Agent
    selects which ones to send to the Validation Agents.
    """
    hypothesis_bug_id: str       # the DiagnosticHypothesis this addresses
    diff: str                    # unified diff string (patch -p1 compatible)
    strategy: RepairStrategy
    source: str                  # "rules" | "llm" | "memory" | "hybrid"
    confidence: float            # agent's self-estimated quality, 0..1
    changed_files: List[str] = field(default_factory=list)
    changed_lines: int = 0       # total lines changed (size proxy)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_bug_id": self.hypothesis_bug_id,
            "diff": self.diff,
            "strategy": self.strategy.value,
            "source": self.source,
            "confidence": self.confidence,
            "changed_files": self.changed_files,
            "changed_lines": self.changed_lines,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatchCandidate":
        d = dict(d)
        d["strategy"] = RepairStrategy(d.get("strategy", "unknown"))
        return cls(**d)


# ---------------------------------------------------------------------------
# ValidationVerdict  (emitted by Validation Agents — Person 3)
# ---------------------------------------------------------------------------

@dataclass
class ValidationVerdict:
    """Final verdict from the Empirical Validation Agents.

    Person 3's One-Batch Overfit agent and Multi-Seed Statistical agent
    both emit this.  The Supervisor uses `accepted` to decide whether to
    commit the patch and store the trajectory in the Experience Memory.
    """
    patch_candidate: PatchCandidate
    accepted: bool
    validator: str               # "overfit_oracle" | "multiseed_stats"
    p_value: Optional[float] = None      # from KS / t-test, if available
    effect_size: Optional[float] = None  # paired improvement over baseline
    seeds_run: int = 0
    compute_budget_used: float = 0.0     # fraction of allowed compute
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_candidate": self.patch_candidate.to_dict(),
            "accepted": self.accepted,
            "validator": self.validator,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "seeds_run": self.seeds_run,
            "compute_budget_used": self.compute_budget_used,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidationVerdict":
        d = dict(d)
        d["patch_candidate"] = PatchCandidate.from_dict(d["patch_candidate"])
        return cls(**d)


# ---------------------------------------------------------------------------
# AgentState  (the LangGraph / supervisor shared state node)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Shared state object threaded through every LangGraph node.

    The Supervisor Agent reads and writes this; each specialist agent
    receives the *full* state and appends to the relevant list fields.
    Immutable fields (target_file, run_id) are set at pipeline entry.

    Field ownership
    ---------------
    target_file, run_id         -- set by the CLI / entry point
    detection_report            -- written by Person 2 (Rishikesh)
    hypotheses                  -- written by Person 4 (DiagnosisAgent)
    patch_candidates            -- written by Person 4 (FixSynthesizerAgent)
    validation_verdicts         -- written by Person 3 (Validation agents)
    experience_ids              -- written by Person 1 (Memory agent)
    status, abort_reason        -- managed by Person 1 (Supervisor)
    """

    # -- Immutable context --------------------------------------------------
    run_id: str = ""
    target_file: str = ""                    # the script being diagnosed

    # -- Detection (Person 2 / Rishikesh) -----------------------------------
    detection_report: Optional[Report] = None

    # -- Diagnosis (Person 4) -----------------------------------------------
    hypotheses: List[DiagnosticHypothesis] = field(default_factory=list)

    # -- Repair / Fix (Person 4) --------------------------------------------
    patch_candidates: List[PatchCandidate] = field(default_factory=list)

    # -- Validation (Person 3) ----------------------------------------------
    validation_verdicts: List[ValidationVerdict] = field(default_factory=list)

    # -- Memory (Person 1) --------------------------------------------------
    experience_ids: List[str] = field(default_factory=list)

    # -- Supervisor bookkeeping (Person 1) ----------------------------------
    status: AgentStatus = AgentStatus.PENDING
    abort_reason: str = ""
    iteration: int = 0       # repair-loop retry counter
    max_iterations: int = 3

    # -- Pass-through metadata ----------------------------------------------
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience helpers ------------------------------------------------

    @property
    def accepted_patches(self) -> List[ValidationVerdict]:
        return [v for v in self.validation_verdicts if v.accepted]

    @property
    def best_hypothesis(self) -> Optional[DiagnosticHypothesis]:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)

    def should_abort(self) -> bool:
        return (
            self.status == AgentStatus.ABORTED
            or self.iteration >= self.max_iterations
        )
