"""Test fixed-point solvers."""

import numpy as np
import pytest
from numba import cfunc, float64, int64, njit

from scarfs import (
    hypercube_fixed_point,
    labeled_subsimplex,
    simplex_fixed_point,
    simplotope_fixed_point,
)


@cfunc(float64[:](float64[:]))
def roll_func(simp: np.ndarray) -> np.ndarray:
    """Roll the simplex coordinates by one position."""
    return np.roll(simp, 1)


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_roll(dim: int, rng: np.random.Generator) -> None:
    """Test the fixed point of a rolling map on the simplex."""
    start = rng.random(dim)
    start /= start.sum()
    res = simplex_fixed_point(roll_func, start, 100)
    assert np.all(res >= 0)
    assert np.isclose(res.sum(), 1)
    assert np.allclose(res, 1 / dim, atol=0.01)


def test_rps_fixed_point(rng: np.random.Generator) -> None:
    """Test the fixed point of a rock-paper-scissors dynamic."""
    start = rng.random(3)
    start /= start.sum()
    weights = 1 + 3 * rng.random(3)
    vara, varb, varc = weights
    expected = np.linalg.solve(
        [[0, -vara, 1, 1], [1, 0, -varb, 1], [-varc, 1, 0, 1], [1, 1, 1, 0]],
        [0, 0, 0, 1],
    )[:-1]

    @njit(float64[:](float64[::1]))
    def func(inp: np.ndarray) -> np.ndarray:
        """Apply one step of the replicator dynamic."""
        inter = np.roll(inp, 1) - weights * np.roll(inp, -1)
        res = np.maximum(0, inter - inter @ inp) + inp
        return res / res.sum()

    res = simplex_fixed_point(func, start, 1000)
    assert np.allclose(res, expected, atol=5e-3)


_TRANS = np.array([0.5, 0.5])
_ROT = np.array([[0, 1], [-1, 0]], float)


@cfunc(float64[:](float64[:]))
def rotate(inp: np.ndarray) -> np.ndarray:
    """Rotate a point ninety degrees about the hypercube center."""
    return (inp - _TRANS) @ _ROT + _TRANS


def test_rotate(rng: np.random.Generator) -> None:
    """Test the fixed point of a rotation on the hypercube."""
    start = rng.random(2)
    res = hypercube_fixed_point(rotate, start, 100)
    assert np.all(res >= 0)
    assert np.all(res <= 1)
    assert np.allclose(res, 0.5, atol=0.05)


@cfunc(float64[:](float64[:]))
def rolltate(inp: np.ndarray) -> np.ndarray:
    """Roll the first simplex and rotate the remaining simplotope factors."""
    res = np.empty(7)
    res[:3] = np.roll(inp[:3], 1)
    new_rot = (inp[3::2] - _TRANS) @ _ROT + _TRANS
    res[3::2] = new_rot
    res[4::2] = 1 - new_rot
    return res


def test_rolltate(rng: np.random.Generator) -> None:
    """Test the fixed point of a roll-and-rotate map on a simplotope."""
    runs = np.array([3, 2, 2])
    gaps = np.insert(runs.cumsum(), 0, 0)
    start = rng.random(7)
    start /= np.add.reduceat(start, gaps[:-1]).repeat(runs)
    res = simplotope_fixed_point(rolltate, start, runs, 100)
    expected = np.array([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 0.5, 0.5])
    assert np.allclose(res, expected, atol=0.05)


@cfunc(int64(float64[::1]))
def invalid(simp: np.ndarray) -> int:
    """Return an improper label that ignores some simplex vertices."""
    for i in range(simp.size):
        if simp[i] < 1e-3:  # noqa: PLR2004
            return i
    return 0


@cfunc(int64(float64[::1]))
def valid(simp: np.ndarray) -> int:
    """Return a proper label for the first positive coordinate."""
    for i in range(simp.size):
        if simp[i] > 0:
            return i
    return 0


def test_improper_label_function(rng: np.random.Generator) -> None:
    """Test that improper label functions raise an error."""
    start = rng.random(4)
    start /= start.sum()
    with pytest.raises(ValueError):
        labeled_subsimplex(invalid, start, 100)
    with pytest.raises(ValueError):
        labeled_subsimplex(valid, -start, 100)
    with pytest.raises(ValueError):
        labeled_subsimplex(valid, start, 1)


@cfunc(float64[::1](float64[::1]))
def to_vertex(simp: np.ndarray) -> np.ndarray:
    """Map every point to the first vertex of the simplex."""
    res = np.zeros_like(simp)
    res[0] = 1.0
    return res


def test_disc_two_boundary() -> None:
    """Test the solver runs at the minimum discretization of two."""
    res = simplex_fixed_point(roll_func, np.full(3, 1 / 3), 2)
    assert np.all(res >= 0)
    assert np.isclose(res.sum(), 1)
    assert np.allclose(res, 1 / 3, atol=0.01)


def test_vertex_fixed_point() -> None:
    """Test a fixed point that lies on a vertex of the simplex."""
    res = simplex_fixed_point(to_vertex, np.full(4, 0.25), 50)
    expected = np.zeros(4)
    expected[0] = 1
    assert np.allclose(res, expected, atol=0.05)


def test_warm_start_refinement() -> None:
    """Test that warm starting at a finer discretization refines the result."""
    start = np.full(4, 0.25)
    coarse = simplex_fixed_point(roll_func, start, 20)
    fine = simplex_fixed_point(roll_func, coarse / coarse.sum(), 200)
    assert np.max(np.abs(fine - 0.25)) < np.max(np.abs(coarse - 0.25))
    assert np.allclose(fine, 0.25, atol=0.01)


def test_invalid_inputs() -> None:
    """Test that the public solvers reject malformed inputs."""
    simplex = np.full(3, 1 / 3)
    with pytest.raises(ValueError):
        simplex_fixed_point(roll_func, simplex, 1)
    with pytest.raises(ValueError):
        simplex_fixed_point(roll_func, np.full(3, 1.0), 100)
    with pytest.raises(ValueError):
        hypercube_fixed_point(rotate, np.array([2.0, 0.0]), 100)
    with pytest.raises(ValueError):
        simplotope_fixed_point(rolltate, np.full(2, 0.5), np.array([0, 2]), 100)
    with pytest.raises(ValueError):
        simplotope_fixed_point(rolltate, simplex, np.array([2, 2]), 100)
    with pytest.raises(ValueError):
        simplotope_fixed_point(
            rolltate, np.array([0.5, 0.5, 0.3, 0.3]), np.array([2, 2]), 100
        )
