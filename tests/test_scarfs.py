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
def test_roll(dim: int) -> None:
    """Test the fixed point of a rolling map on the simplex."""
    start = np.random.rand(dim)
    start /= start.sum()
    res = simplex_fixed_point(roll_func, start, 100)
    assert np.all(res >= 0)
    assert np.isclose(res.sum(), 1)
    assert np.allclose(res, 1 / dim, atol=0.01)


def test_rps_fixed_point() -> None:
    """Test the fixed point of a rock-paper-scissors dynamic."""
    start = np.random.rand(3)
    start /= start.sum()
    weights = 1 + 3 * np.random.random(3)
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


def test_rotate() -> None:
    """Test the fixed point of a rotation on the hypercube."""
    start = np.random.rand(2)
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


def test_rolltate() -> None:
    """Test the fixed point of a roll-and-rotate map on a simplotope."""
    runs = np.array([3, 2, 2])
    gaps = np.insert(runs.cumsum(), 0, 0)
    start = np.random.rand(7)
    start /= np.add.reduceat(start, gaps[:-1]).repeat(runs)
    res = simplotope_fixed_point(rolltate, start, runs, 100)
    expected = np.array([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 0.5, 0.5])
    assert np.allclose(res, expected, atol=0.05)


@cfunc(int64(float64[:]))
def invalid(simp: np.ndarray) -> int:
    """Return an improper label that ignores some simplex vertices."""
    for i in range(simp.size):
        if simp[i] < 1e-3:  # noqa: PLR2004
            return i
    return 0


@cfunc(int64(float64[:]))
def valid(simp: np.ndarray) -> int:
    """Return a proper label for the first positive coordinate."""
    for i in range(simp.size):
        if simp[i] > 0:
            return i
    return 0


def test_improper_label_function() -> None:
    """Test that improper label functions raise an error."""
    start = np.random.rand(4)
    start /= start.sum()
    with pytest.raises(ValueError):
        labeled_subsimplex(invalid, start, 100)
    with pytest.raises(ValueError):
        labeled_subsimplex(valid, -start, 100)
    with pytest.raises(ValueError):
        labeled_subsimplex(valid, start, 1)
