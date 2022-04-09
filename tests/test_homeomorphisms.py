from itertools import product
import pytest
from scarfs import (
    _hypercube_to_simplex,
    _simplex_to_hypercube,
    _simplotope_to_simplex,
    _simplex_to_simplotope,
    _add_reduceat,
)
import numpy as np


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_reduce_at(simplicies: int) -> None:
    for _ in range(100):
        runs = np.random.randint(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        n = gaps[-1]

        rands = np.arange(n, dtype=float)
        nump = np.add.reduceat(rands, gaps[:-1])
        cust = _add_reduceat(rands, gaps[1:])
        assert np.allclose(nump, cust)


def test_known_hypercube_homeomorphisms() -> None:
    for hyper, simp in [
        ([0, 0], [0, 0, 1]),
        ([1, 1], [0.5, 0.5, 0]),
        ([1, 0, 0], [1, 0, 0, 0]),
    ]:
        ahyper = np.array(hyper)
        asimp = np.array(simp)
        assert np.allclose(_hypercube_to_simplex(ahyper), asimp)
        assert np.allclose(_simplex_to_hypercube(asimp), ahyper)


def assert_hypercube_to_simplex_homeomorphism(hyper: np.ndarray) -> None:
    dim = hyper.size
    simp = _hypercube_to_simplex(hyper)
    assert simp.size == dim + 1
    assert np.all(0 <= simp)
    assert np.isclose(simp.sum(), 1)
    point = _simplex_to_hypercube(simp)
    assert np.allclose(hyper, point)


def assert_simplex_to_hypercube_homeomorphism(simp: np.ndarray) -> None:
    dim = simp.size
    hyper = _simplex_to_hypercube(simp)
    assert hyper.size == dim - 1
    assert np.all(0 <= hyper) and np.all(hyper <= 1)
    point = _hypercube_to_simplex(hyper)
    assert np.allclose(simp, point)


def test_edge_hypercube_homeomorphisms() -> None:
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
def test_random_hypercube_homeomorphism(dim: int) -> None:
    for _ in range(100):
        hyper = np.clip(np.random.rand(dim) * 2 - 0.5, 0, 1)
        assert_hypercube_to_simplex_homeomorphism(hyper)

    for _ in range(100):
        simp = np.random.rand(dim)
        mask = ((np.random.randint(2**dim - 1) + 1) >> np.arange(dim)) % 2
        simp *= mask
        simp /= simp.sum()
        assert_simplex_to_hypercube_homeomorphism(simp)


def assert_simplotope_to_simplex_homeomorphism(
    tope: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> None:
    simp = _simplotope_to_simplex(tope, runs, gaps)
    assert simp.size == gaps[-1] - runs.size + 1
    assert np.all(0 <= simp)
    assert np.isclose(simp.sum(), 1)
    point = _simplex_to_simplotope(simp, runs, gaps)
    assert np.allclose(tope, point)


def assert_simplex_to_simplotope_homeomorphism(
    simp: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> None:
    gaps = np.insert(runs.cumsum(), 0, 0)
    tope = _simplex_to_simplotope(simp, runs, gaps)
    assert tope.size == gaps[-1]
    assert np.all(0 <= tope)
    assert np.allclose(np.add.reduceat(tope, gaps[:-1]), 1)
    point = _simplotope_to_simplex(tope, runs, gaps)
    assert np.allclose(simp, point)


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_random_simplotope_homeomorphism(simplicies: int) -> None:
    for _ in range(100):
        runs = np.random.randint(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        tope = np.random.rand(gaps[-1])
        tope /= np.add.reduceat(tope, gaps[:-1]).repeat(runs)
        assert_simplotope_to_simplex_homeomorphism(tope, runs, gaps)

    for _ in range(100):
        runs = np.random.randint(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        dim = gaps[-1] - simplicies + 1
        simp = np.random.rand(dim)
        mask = ((np.random.randint(2**dim - 1) + 1) >> np.arange(dim)) % 2
        simp *= mask
        simp /= simp.sum()
        assert_simplex_to_simplotope_homeomorphism(simp, runs, gaps)


@pytest.mark.parametrize("simplicies", [1, 2, 3, 7])
def test_edge_simplotope_homeomorphisms(simplicies: int) -> None:
    for _ in range(10):
        runs = np.random.randint(1, 4, simplicies)
        gaps = np.insert(runs.cumsum(), 0, 0)
        simp_dim = gaps[-1] - runs.size + 1

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
def test_hypercube_specialization_of_simplotope(dim: int) -> None:
    for _ in range(100):
        runs = np.full(dim, 2)
        gaps = np.arange(0, 2 * dim + 1, 2)

        hyper = np.random.rand(dim)
        tope = np.empty(dim * 2)
        tope[::2] = hyper
        tope[1::2] = 1 - hyper
        hsimp = _hypercube_to_simplex(hyper)
        tsimp = _simplotope_to_simplex(tope, runs, gaps)
        assert np.allclose(hsimp, tsimp)

        simp = np.random.rand(dim + 1)
        simp /= simp.sum()
        shyper = _simplex_to_hypercube(simp)
        stope = _simplex_to_simplotope(simp, runs, gaps)
        assert np.allclose(shyper, stope[::2])
