"""Benchmark the fixed-point solvers."""

import numpy as np
import pytest
from numba import cfunc, float64
from pytest_benchmark.fixture import BenchmarkFixture

from scarfs import simplex_fixed_point


@cfunc(float64[::1](float64[::1]))
def _roll(simp: np.ndarray) -> np.ndarray:
    """Roll the simplex by one, whose only fixed point is the uniform point."""
    return np.roll(simp, 1)


@pytest.mark.long
@pytest.mark.parametrize(("dim", "disc"), [(3, 100), (5, 200), (10, 200)])
def test_simplex_fixed_point_benchmark(
    benchmark: BenchmarkFixture, dim: int, disc: int
) -> None:
    """Benchmark a repeated solve, which should not recompile each call."""
    start = np.full(dim, 1 / dim)
    simplex_fixed_point(_roll, start, disc)  # warm up the jit compilation
    result = benchmark(simplex_fixed_point, _roll, start, disc)
    assert np.allclose(result, 1 / dim, atol=0.02)
