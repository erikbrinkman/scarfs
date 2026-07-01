"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a freshly seeded RNG so randomized tests are deterministic."""
    return np.random.default_rng(0)
