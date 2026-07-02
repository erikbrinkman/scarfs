"""Benchmark the fixed-point solver under its two real usage patterns.

The solve itself is microsecond-scale, so these benchmarks instead measure the cost
of a whole *usage* with a freshly defined function -- where the solver's compilation
strategy actually shows up: a one-shot solve, and a four-step warm-start refinement.
Each round is set up with a fresh function (so the setup, not the timed body, pays any
per-function compilation), which keeps a healthy solver sub-millisecond and would make
a regression that recompiles the pivot routine per function balloon into hundreds.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numba import float64, jit
from numpy.typing import NDArray
from pytest_benchmark.fixture import BenchmarkFixture

from scarfs import simplex_fixed_point

_SimplexMap = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_DIM = 5


def _displaced_simplex(dim: int) -> NDArray[np.float64]:
    """Build a point displaced from the uniform fixed point so the pivot walks."""
    point = np.arange(1, dim + 1, dtype=np.float64)
    return point / point.sum()


_START = _displaced_simplex(_DIM)


def _fresh_roll() -> _SimplexMap:
    """Build a freshly compiled roll map, as defining a new problem would."""

    @jit(float64[::1](float64[::1]))
    def roll(simp: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover
        return np.roll(simp, 1)

    roll(_START.copy())
    return roll


def _setup() -> tuple[tuple[_SimplexMap], dict[str, object]]:
    return ((_fresh_roll(),), {})


@pytest.mark.long
def test_single_use(benchmark: BenchmarkFixture) -> None:
    """Benchmark one solve per distinct function, the one-shot usage pattern."""

    def solve(func: _SimplexMap) -> NDArray[np.float64]:
        return simplex_fixed_point(func, _START, 200)

    result: NDArray[np.float64] = benchmark.pedantic(solve, setup=_setup, rounds=20)
    assert np.allclose(result, 1 / _DIM, atol=0.02)


@pytest.mark.long
def test_iterative_refinement(benchmark: BenchmarkFixture) -> None:
    """Benchmark a four-step warm-start refinement, the recommended usage pattern."""

    def refine(func: _SimplexMap) -> NDArray[np.float64]:
        point = _START
        for disc in (50, 100, 200, 400):
            point = simplex_fixed_point(func, point / point.sum(), disc)
        return point

    result: NDArray[np.float64] = benchmark.pedantic(refine, setup=_setup, rounds=20)
    assert np.allclose(result, 1 / _DIM, atol=0.02)
