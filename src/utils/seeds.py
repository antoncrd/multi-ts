"""Reproducibility utilities for seed management."""

import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SeedSequence:
    """Derive deterministic child seeds for parallel runs."""

    def __init__(self, base_seed: int):
        self._rng = np.random.SeedSequence(base_seed)

    def spawn(self, n: int) -> list[int]:
        """Return n deterministic child seeds."""
        children = self._rng.spawn(n)
        return [int(c.generate_state(1)[0]) for c in children]
