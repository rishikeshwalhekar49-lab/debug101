"""Weight inspection: a read-only look at a model's numbers.

It reports anomalies in trained weights (non-finite or exploded values, zeroed
matrices, collapsed biases and rank). It does not edit weights: a repair belongs in
the training code, where `fix` can test it against the validation gates.
"""

from .inspector import WeightInspector, inspect_script
from .schemas import WeightAnomalyType, WeightFinding, WeightReport

__all__ = ["WeightAnomalyType", "WeightFinding", "WeightReport", "WeightInspector", "inspect_script"]
