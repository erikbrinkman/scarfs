"""Test homeomorphisms."""

from itertools import product
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scarfs._homeomorphisms import (
    add_reduceat,
    hypercube_to_simplex,
    simplex_to_hypercube,
    simplex_to_simplotope,
    simplotope_to_simplex,
)


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_reduce_at(simplicies: int, rng: np.random.Generator) -> None:
    """Test that the custom reduce-at matches numpy's."""
    for _ in range(100):
        runs = rng.integers(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        n = gaps[-1]

        rands = np.arange(n, dtype=float)
        nump = np.add.reduceat(rands, gaps[:-1])
        cust = add_reduceat(rands, gaps[1:])
        assert np.allclose(nump, cust)


def test_known_hypercube_homeomorphisms() -> None:
    """Test hypercube-simplex maps on known point pairs."""
    for hyper, simp in [
        ([0, 0], [0, 0, 1]),
        ([1, 1], [0.5, 0.5, 0]),
        ([1, 0, 0], [1, 0, 0, 0]),
    ]:
        ahyper = np.array(hyper, "f8")
        asimp = np.array(simp, "f8")
        assert np.allclose(hypercube_to_simplex(ahyper), asimp)
        assert np.allclose(simplex_to_hypercube(asimp), ahyper)


def assert_hypercube_to_simplex_homeomorphism(hyper: np.ndarray) -> None:
    """Assert the hypercube-to-simplex map round-trips for a point."""
    dim = hyper.size
    simp = hypercube_to_simplex(hyper)
    assert simp.size == dim + 1
    assert np.all(0 <= simp)
    assert np.isclose(simp.sum(), 1)
    point = simplex_to_hypercube(simp)
    assert np.allclose(hyper, point)


def assert_simplex_to_hypercube_homeomorphism(simp: np.ndarray) -> None:
    """Assert the simplex-to-hypercube map round-trips for a point."""
    dim = simp.size
    hyper = simplex_to_hypercube(simp)
    assert hyper.size == dim - 1
    assert np.all(0 <= hyper) and np.all(hyper <= 1)
    point = hypercube_to_simplex(hyper)
    assert np.allclose(simp, point)


def test_edge_hypercube_homeomorphisms() -> None:
    """Test hypercube-simplex maps on edge and midpoint cases."""
    # edges
    for lpoints in product(*([[0, 1]] * 3)):
        hyper = np.array(lpoints, float)
        assert_hypercube_to_simplex_homeomorphism(hyper)

    # mid point
    assert_hypercube_to_simplex_homeomorphism(np.full(4, 0.5))

    # simplex edges
    for simp in np.eye(5):
        assert_simplex_to_hypercube_homeomorphism(simp)

    # mid point
    assert_simplex_to_hypercube_homeomorphism(np.full(4, 1 / 4))


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_random_hypercube_homeomorphism(dim: int, rng: np.random.Generator) -> None:
    """Test hypercube-simplex maps on random points."""
    for _ in range(100):
        hyper = np.clip(rng.random(dim) * 2 - 0.5, 0, 1)
        assert_hypercube_to_simplex_homeomorphism(hyper)

    for _ in range(100):
        simp = rng.random(dim)
        mask: NDArray[np.int_] = (
            (cast(int, rng.integers(2**dim - 1)) + 1) >> np.arange(dim)
        ) % 2
        simp *= mask
        simp /= simp.sum()
        assert_simplex_to_hypercube_homeomorphism(simp)


def assert_simplotope_to_simplex_homeomorphism(
    tope: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> None:
    """Assert the simplotope-to-simplex map round-trips for a point."""
    simp = simplotope_to_simplex(tope, runs, gaps)
    assert simp.size == gaps[-1] - runs.size + 1
    assert np.all(0 <= simp)
    assert np.isclose(simp.sum(), 1)
    point = simplex_to_simplotope(simp, runs, gaps)
    assert np.allclose(tope, point)


def assert_simplex_to_simplotope_homeomorphism(
    simp: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> None:
    """Assert the simplex-to-simplotope map round-trips for a point."""
    gaps = np.insert(runs.cumsum(), 0, 0)
    tope = simplex_to_simplotope(simp, runs, gaps)
    assert tope.size == gaps[-1]
    assert np.all(0 <= tope)
    assert np.allclose(np.add.reduceat(tope, gaps[:-1]), 1)
    point = simplotope_to_simplex(tope, runs, gaps)
    assert np.allclose(simp, point)


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_random_simplotope_homeomorphism(
    simplicies: int, rng: np.random.Generator
) -> None:
    """Test simplotope-simplex maps on random points."""
    for _ in range(100):
        runs = rng.integers(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        tope = rng.random(int(gaps[-1]))
        tope /= np.add.reduceat(tope, gaps[:-1]).repeat(runs)
        assert_simplotope_to_simplex_homeomorphism(tope, runs, gaps)

    for _ in range(100):
        runs = rng.integers(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        dim: int = gaps[-1] - simplicies + 1
        simp = rng.random(dim)
        mask: NDArray[np.int_] = (
            (cast(int, rng.integers(2**dim - 1)) + 1) >> np.arange(dim)
        ) % 2
        simp *= mask
        simp /= simp.sum()
        assert_simplex_to_simplotope_homeomorphism(simp, runs, gaps)


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_edge_simplotope_homeomorphisms(
    simplicies: int, rng: np.random.Generator
) -> None:
    """Test simplotope-simplex maps on edge and midpoint cases."""
    for _ in range(10):
        runs = rng.integers(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        simp_dim: int = gaps[-1] - runs.size + 1

        # edges
        for topes in product(*(np.eye(v) for v in runs)):
            tope = np.concatenate(topes)
            assert_simplotope_to_simplex_homeomorphism(tope, runs, gaps)

        # mid point
        assert_simplotope_to_simplex_homeomorphism((1 / runs).repeat(runs), runs, gaps)

        # simplex edges
        for simp in np.eye(simp_dim):
            assert_simplex_to_simplotope_homeomorphism(simp, runs, gaps)

        # mid point
        assert_simplex_to_simplotope_homeomorphism(
            np.full(simp_dim, 1 / simp_dim), runs, gaps
        )


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_hypercube_specialization_of_simplotope(
    dim: int, rng: np.random.Generator
) -> None:
    """Test that the hypercube maps specialize the simplotope maps."""
    for _ in range(100):
        runs = np.full(dim, 2)
        gaps = np.arange(0, 2 * dim + 1, 2)

        hyper = rng.random(dim)
        tope = np.empty(dim * 2)
        tope[::2] = hyper
        tope[1::2] = 1 - hyper
        hsimp = hypercube_to_simplex(hyper)
        tsimp = simplotope_to_simplex(tope, runs, gaps)
        assert np.allclose(hsimp, tsimp)

        simp = rng.random(dim + 1)
        simp /= simp.sum()
        shyper = simplex_to_hypercube(simp)
        stope = simplex_to_simplotope(simp, runs, gaps)
        assert np.allclose(shyper, stope[::2])
