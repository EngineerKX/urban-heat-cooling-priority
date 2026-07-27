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
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
