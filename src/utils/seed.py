"""One place to seed every RNG the pipeline touches. RANDOM_SEED=42 was
pinned across every notebook individually ("per eval rules") — set it once
here instead."""

import random

import numpy as np

from config.settings import RANDOM_SEED


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
