"""Alias module to support multiseed_agent naming convention."""

from .multiseedagent import (
    DistributionComparisonResult,
    MultiseedAgent,
    apply_patch_string,
    compare_distributions,
    compute_cohens_d,
    compute_confidence_interval,
)

__all__ = [
    "MultiseedAgent",
    "DistributionComparisonResult",
    "compare_distributions",
    "compute_confidence_interval",
    "compute_cohens_d",
    "apply_patch_string",
]
