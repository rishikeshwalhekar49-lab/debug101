"""Loads and validates bug_taxonomy.yaml into BugPattern objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

_REQUIRED_FIELDS = {
    "id",
    "name",
    "category",
    "symptoms",
    "invariant_ids",
    "static_detectable",
    "runtime_detectable",
    "semantic_detectable",
    "typical_fix",
}
_VALID_CATEGORIES = {"data", "model_logic", "training_dynamics", "api_misuse", "environment"}

_DEFAULT_PATH = Path(__file__).parent / "bug_taxonomy.yaml"


@dataclass
class BugPattern:
    id: str
    name: str
    category: str
    symptoms: List[str]
    invariant_ids: List[str]
    static_detectable: bool
    runtime_detectable: bool
    semantic_detectable: bool
    typical_fix: str


class TaxonomyValidationError(ValueError):
    pass


def load_taxonomy(path: Path = _DEFAULT_PATH) -> List[BugPattern]:
    """Load and validate the taxonomy YAML, returning BugPattern objects.

    Raises TaxonomyValidationError with every problem found (not just the
    first) so a contributor adding a new entry gets complete feedback in
    one pass.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise TaxonomyValidationError("Taxonomy file must contain a top-level list of entries")

    errors = []
    seen_ids = set()
    patterns = []

    for i, entry in enumerate(raw):
        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"Entry {i} ({entry.get('id', '?')}): missing fields {sorted(missing)}")
            continue

        if entry["id"] in seen_ids:
            errors.append(f"Entry {i}: duplicate id '{entry['id']}'")
        seen_ids.add(entry["id"])

        if entry["category"] not in _VALID_CATEGORIES:
            errors.append(
                f"Entry {entry['id']}: invalid category '{entry['category']}', "
                f"must be one of {sorted(_VALID_CATEGORIES)}"
            )
            continue

        patterns.append(
            BugPattern(
                id=entry["id"],
                name=entry["name"],
                category=entry["category"],
                symptoms=list(entry["symptoms"]),
                invariant_ids=list(entry["invariant_ids"]),
                static_detectable=bool(entry["static_detectable"]),
                runtime_detectable=bool(entry["runtime_detectable"]),
                semantic_detectable=bool(entry["semantic_detectable"]),
                typical_fix=entry["typical_fix"],
            )
        )

    if errors:
        raise TaxonomyValidationError("Taxonomy validation failed:\n  " + "\n  ".join(errors))

    return patterns


def coverage_report(patterns: List[BugPattern]) -> dict:
    """Summarize how many taxonomy entries already have a runtime invariant.

    Useful for prioritizing Phase 1/2 work: entries with
    runtime_detectable=True but invariant_ids=[] are gaps in this
    package's implementation, not gaps in the taxonomy itself.
    """
    total = len(patterns)
    with_invariant = sum(1 for p in patterns if p.invariant_ids)
    gaps = [p.id for p in patterns if p.runtime_detectable and not p.invariant_ids]
    return {
        "total_patterns": total,
        "patterns_with_invariant": with_invariant,
        "runtime_detectable_gaps": gaps,
    }
