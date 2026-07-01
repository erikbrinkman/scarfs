"""Homeomorphisms between bounded spaces and the simplex."""

import numpy as np
from numba import cfunc, float64, int64, types
from numpy.typing import NDArray


@cfunc(float64[::1](float64[:], types.Array(int64, 1, "A", readonly=True)), cache=True)
def add_reduceat(
    arr: NDArray[np.float64], inds: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute a numba version of ``np.add.reduceat``.

    Note that this doesn't quite operate like reduceat, but accomplishes the
    same thing for these call sites. In contrast to reduceat where you pass
    the start of slices, here you pass the end.
    """
    res = np.zeros(inds.size)
    j = 0
    for i, v in enumerate(arr):
        while inds[j] <= i:
            j += 1
        res[j] += v
    return res


@cfunc(
    float64[:](
        float64[:],
        types.Array(int64, 1, "A", readonly=True),
        types.Array(int64, 1, "A", readonly=True),
    ),
    cache=True,
)
def simplotope_to_simplex(
    tope: NDArray[np.float64], runs: NDArray[np.int64], gaps: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Map the simplotope to the simplex."""
    resid = 1 - tope[gaps[1:] - 1]
    prop = np.max(resid)
    simp = np.empty((gaps[-1] - runs.size + 1,))
    simp[-1] = 1 - prop
    if prop == 0.0:
        simp[:-1].fill(0)
    else:
        simp[:-1] = np.delete(tope, gaps[1:] - 1) * prop / resid.sum()
    return simp


@cfunc(
    float64[:](
        float64[:],
        types.Array(int64, 1, "A", readonly=True),
        types.Array(int64, 1, "A", readonly=True),
    )
)
def simplex_to_simplotope(
    simp: NDArray[np.float64], runs: NDArray[np.int64], gaps: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Map the simplex to the simplotope."""
    prop = 1 - simp[-1]
    tope = np.zeros(gaps[-1])
    if prop != 0:
        tope[np.delete(np.arange(gaps[-1], dtype=np.int64), gaps[1:] - 1)] = simp[:-1]
        tope *= prop / add_reduceat(tope, gaps[1:]).max()
    tope[gaps[1:] - 1] = 1 - add_reduceat(tope, gaps[1:])
    return np.maximum(tope, 0)


@cfunc(float64[:](float64[:]), cache=True)
def hypercube_to_simplex(hyper: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map the unit hypercube to the simplex."""
    prop = np.max(hyper)
    simp = np.empty((hyper.size + 1,))
    simp[-1] = 1 - prop
    if prop == 0.0:
        simp[:-1].fill(0)
    else:
        simp[:-1] = hyper * prop / hyper.sum()
    return simp


@cfunc(float64[:](float64[:]), cache=True)
def simplex_to_hypercube(simp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map the simplex to the unit hypercube."""
    prop = 1 - simp[-1]
    if prop == 0:
        return np.zeros(simp.size - 1)
    else:
        return simp[:-1] * prop / simp[:-1].max()
