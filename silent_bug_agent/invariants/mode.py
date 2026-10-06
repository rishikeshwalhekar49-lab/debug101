from .base import Invariant, InvariantResult, Severity, TrainingState

# Module name substrings whose train/eval mode changes real behavior
# (BatchNorm uses running stats vs batch stats; Dropout turns off).
# Extend this list as new mode-sensitive layer types come up.
_MODE_SENSITIVE_SUBSTRINGS = ("batchnorm", "dropout", "instancenorm")


class TrainEvalModeInvariant(Invariant):
    """Flags mode-sensitive modules left in the wrong train/eval state.

    Catches the classic silent bug where model.eval() was never called
    before validation (BatchNorm keeps using batch statistics, Dropout
    stays active), or model.train() was never restored afterward.
    """

    id = "train_eval_mode_mismatch"
    description = "BatchNorm/Dropout-like modules must match the current train/eval phase."

    def check(self, state: TrainingState) -> InvariantResult:
        expected_mode = "eval" if state.is_eval_phase else "train"
        offenders = {
            name: mode
            for name, mode in state.module_modes.items()
            if any(s in name.lower() for s in _MODE_SENSITIVE_SUBSTRINGS) and mode != expected_mode
        }
        if not offenders:
            return InvariantResult(self.id, True, Severity.INFO, "Module modes match the current module")

        return InvariantResult(
            self.id,
            False,
            Severity.CRITICAL,
            f"{len(offenders)} mode-sensitive module(s) not in '{expected_mode}' mode "
            f"during {'eval' if state.is_eval_phase else 'training'}",
            {"offenders": offenders, "step": state.step},
        )
