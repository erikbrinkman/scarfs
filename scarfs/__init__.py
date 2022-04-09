from numba import njit, float64, int64, jit, cfunc, types
from typing import Callable
import numpy as np


@cfunc(float64[::1](float64[:], types.Array(int64, 1, "A", readonly=True)))
def _add_reduceat(arr: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """numba version of np.add.reduceat

    Note that this doesn't quite operate like reduceat, but accomplishes the
    same thing for these call sites. In constrast to reduceat where you pass
    the start of slices, here you pass the end.
    """
    res = np.zeros(inds.size)
    j = 0
    for i, v in enumerate(arr):
        while inds[j] <= i:
            j += 1
        res[j] += v
    return res


def simplotope_fixed_point(
    func: Callable[[np.ndarray], np.ndarray],
    init: np.ndarray,
    runs: np.ndarray,
    disc: int,
) -> np.ndarray:
    """Compute an approximate fixed point of a function on the simplotope

    Each simplex should be stacked next to each other, and runs specifies the
    number of elements in each simplex.

    Parameters
    ----------
    func : A continuous function mapping from the d-simplotope (specified by
        runs) to itself.
    init : An initial guess for the fixed point. Since many may exist, the
        choice of starting point will affect the solution.
    runs : the length of each simplotope in order.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal this much. Since this function computes a homeomorphism to
        the simplex, the distortion can be up to the number of simplicies.
        Therefore if you want a true approximation discritization of `disc`,
        you should specify `disc * runs.size`.
    """

    gaps = np.insert(runs.cumsum(), 0, 0)

    @njit(float64[:](float64[::1]))
    def simplotope_func(simp: np.ndarray) -> np.ndarray:  # pragma: no cover
        return _simplotope_to_simplex(
            func(_simplex_to_simplotope(simp, runs, gaps)), runs, gaps
        )

    return _simplex_to_simplotope(
        simplex_fixed_point(
            simplotope_func, _simplotope_to_simplex(init, runs, gaps), disc
        ),
        runs,
        gaps,
    )


@cfunc(
    float64[:](
        float64[:],
        types.Array(int64, 1, "A", readonly=True),
        types.Array(int64, 1, "A", readonly=True),
    )
)
def _simplotope_to_simplex(
    tope: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> np.ndarray:
    """homeomorphism between the simplotope and the simplex"""
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
def _simplex_to_simplotope(
    simp: np.ndarray, runs: np.ndarray, gaps: np.ndarray
) -> np.ndarray:
    """homeomorphism between simplex and simplotope"""
    prop = 1 - simp[-1]
    tope = np.zeros(gaps[-1])
    if prop != 0:
        tope[np.delete(np.arange(gaps[-1]), gaps[1:] - 1)] = simp[:-1]
        tope *= prop / _add_reduceat(tope, gaps[1:]).max()
    tope[gaps[1:] - 1] = 1 - _add_reduceat(tope, gaps[1:])
    return np.maximum(tope, 0)


def hypercube_fixed_point(
    func: Callable[[np.ndarray], np.ndarray], init: np.ndarray, disc: int
) -> np.ndarray:
    """Compute an approximate fixed point of a function on the unit hypercube

    This function maps the unit hypercube to the simplex, and in so doing
    distorts the space. The worst distortion is at the tip of the hypercube [1,
    1, ..., 1] whic has O(d) distortion, so the discritization will be off by
    O(d) in that case.

    Note this homeomorphism is more efficient than using simplotope_fixed_point
    where each simplex is a 2-simplex.

    Parameters
    ----------
    func : A continuous function mapping from the d-dimensional unit hypercube
        to itself.
    init : An initial guess for the fixed point. Since many may exist, the
        choice of starting point will affect the solution.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal of this much. Note that due to distortion from the
        homeomorphism, this won't accurately reflect the approximation at all
        areads of the hypercube.
    """

    @njit(float64[:](float64[::1]))
    def simplex_func(simp: np.ndarray) -> np.ndarray:  # pragma: no cover
        return _hypercube_to_simplex(func(_simplex_to_hypercube(simp)))

    return _simplex_to_hypercube(
        simplex_fixed_point(simplex_func, _hypercube_to_simplex(init), disc)
    )


@cfunc(float64[:](float64[:]))
def _hypercube_to_simplex(hyper: np.ndarray) -> np.ndarray:
    """homeomorphism between the unit hypercube and the simplex"""
    prop = np.max(hyper)
    simp = np.empty((hyper.size + 1,))
    simp[-1] = 1 - prop
    if prop == 0.0:
        simp[:-1].fill(0)
    else:
        simp[:-1] = hyper * prop / hyper.sum()
    return simp


@cfunc(float64[:](float64[:]))
def _simplex_to_hypercube(simp: np.ndarray) -> np.ndarray:
    """homeomorphism between the simplex and the unit hypercube"""
    prop = 1 - simp[-1]
    if prop == 0:
        return np.zeros(simp.size - 1)
    else:
        return simp[:-1] * prop / simp[:-1].max()


def simplex_fixed_point(
    func: Callable[[np.ndarray], np.ndarray], init: np.ndarray, disc: int
) -> np.ndarray:
    """Compute an approximate fixed point of a function

    Parameters
    ----------
    func : A continuous function mapping from the d-simplex to itself.
    init : An initial guess for the fixed point. Since many may exist, the
        choice of starting point will affect the solution.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal of this much.
    """

    @njit(int64(float64[::1]))
    def fixed_func(simp: np.ndarray) -> int:  # pragma: no cover
        """Labeling function for a fixed point"""
        return np.argmin((simp == 0) - simp + func(simp))

    return labeled_subsimplex(fixed_func, init, disc)


@njit
def labeled_subsimplex(
    label_func: Callable[[np.ndarray], int], init: np.ndarray, disc: int
) -> np.ndarray:  # pragma: no cover
    """Find approximate center of a fully labeled subsimplex

    This runs once at the discretization provided. It is recommended that this
    be run several times with successively finer discretization and warm
    started with the past result.

    Parameters
    ----------
    label_func : A proper lableing function. A labeling function takes an
        element of the d-simplex and returns a label in [0, d). It is proper if
        the label always coresponds to a dimension in support.
    init : An initial guess for where the fully labeled element might be. This
        will be projected onto the simplex if it is not already.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal this much.

    Returns
    -------
    ret : A discretized simplex with 1 coarser resolution (i.e. ret.sum() + 1
        == init.sum()) that is fully labeled.

    Notes
    -----
    This is an implementation of the sandwhich method from [5]_ and [6]_

    .. [5] Kuhn and Mackinnon 1975. Sandwich Method for Finding Fixed Points.
    .. [6] Kuhn 1968. Simplicial Approximation Of Fixed Points.
    """
    if disc < 2:
        raise ValueError("discretization must be at least two")
    elif not np.all(init >= 0) or abs(1 - init.sum()) > 1e-6:
        raise ValueError("must start as a valid simplex")

    dim = init.size
    # Base vertex of the subsimplex currently being used
    dinit = _discretize_mixture(init, disc)
    base = np.append(dinit, 0)
    base[0] += 1
    # permutation array of [1,dim] where v0 = base,
    # v{i+1} = [..., vi_{perms[i] - 1} - 1, vi_{perms[i]} + 1, ...]
    perms = np.arange(1, dim + 1)
    # Array of labels for each vertex
    labels = np.arange(dim + 1)
    labels[dim] = label_func(dinit / disc)
    # Vertex used to label initial vertices (vertex[-1] == 0)
    label_vertex = base[:-1].copy()
    # Last index moved
    index = dim
    # Most recent created index, should be set to
    new_vertex = np.empty((dim + 1,), np.int64)

    while labels[index] < dim:
        # Find duplicate index. this is O(dim) but not a bottleneck
        current_label = labels[index]
        for ind in range(dim + 1):
            if ind != index and labels[ind] == current_label:
                index = ind
                break

        # Flip simplex over at index
        if index == 0:
            base[perms[0]] += 1
            base[perms[0] - 1] -= 1
            perms = np.roll(perms, -1)
            labels = np.roll(labels, -1)
            index = dim

        elif index == dim:
            base[perms[-1] - 1] += 1
            base[perms[-1]] -= 1
            perms = np.roll(perms, 1)
            labels = np.roll(labels, 1)
            index = 0

        else:  # 0 < index < dim
            perms[index - 1], perms[index] = perms[index], perms[index - 1]

        # Compute actual value of flipped vertex
        new_vertex[:] = base
        new_vertex[perms[:index]] += 1
        new_vertex[perms[:index] - 1] -= 1

        if not (np.all(new_vertex >= 0) and new_vertex.sum() == disc + 1):
            raise ValueError("vertex rotation failed, check labeling function")

        # Update label of new vertex
        if new_vertex[-1] == 2:
            labels[index] = dim
        elif new_vertex[-1] == 0:
            labels[index] = np.argmax(new_vertex[:-1] - label_vertex)
        else:  # == 1
            labels[index] = label_func(new_vertex[:-1] / disc)
            if not (0 <= labels[index] < dim and new_vertex[labels[index]]):
                raise ValueError("labeling function was not proper (see help)")

    # Average out all vertices in simplex we care about
    current = base
    if index == 0:  # pragma: no cover
        count = 0
        mean = np.zeros(dim)
    else:  # pragma: no cover
        count = 1
        mean = current.astype(np.float64)
    for i, j in enumerate(perms, 1):
        current[j] += 1
        current[j - 1] -= 1
        if i != index:
            count += 1
            mean += (current - mean) / count
    return mean[:-1] / disc


@cfunc(int64[:](float64[:], int64))
def _discretize_mixture(simp: np.ndarray, k: int) -> np.ndarray:  # pragma: no cover
    """Discretize a mixture

    The returned value will have all integer components that sum to k, with the
    minimum error. Thus, discretizing the mixture.
    """
    disc = np.floor(simp * k).astype(np.int64)
    inds = np.argsort(disc - simp * k)[: k - disc.sum()]
    disc[inds] += 1
    return disc
