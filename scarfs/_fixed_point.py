"""Fixed-point algorithms over the simplex and related spaces."""

from collections.abc import Callable

import numpy as np
from numba import float64, int64, jit, types
from numpy.typing import NDArray

from ._homeomorphisms import (
    hypercube_to_simplex,
    simplex_to_hypercube,
    simplex_to_simplotope,
    simplotope_to_simplex,
)

_SIMPLEX_TOL = 1e-6
# The user's map is passed to the pivot routine as a first-class function so the routine
# compiles a single time for every func and space, rather than recompiling the pivot (or a
# captured relay) per distinct func.
_SIMPLEX_MAP = types.FunctionType(float64[::1](float64[::1]))


@jit(int64[::1](float64[:], int64), cache=True, error_model="numpy")
def discretize_mixture(
    simp: NDArray[np.float64], k: int
) -> NDArray[np.int64]:  # pragma: no cover
    """Discretize a mixture.

    The returned value has all integer components that sum to ``k`` with the
    minimum total rounding error, i.e. it discretizes the mixture onto a
    ``k``-resolution lattice. ``simp`` must already sum to one so that the number
    of components rounded up is always in ``[0, simp.size)``. Ties in the
    fractional remainder are broken by ascending index.
    """
    frac = simp * k
    disc = np.floor(frac).astype(np.int64)
    inds = np.argsort(disc - frac)[: k - disc.sum()]
    disc[inds] += 1
    return disc


@jit(
    int64(_SIMPLEX_MAP, float64[::1], int64[::1], int64[::1]),
    nogil=True,
    error_model="numpy",
)
def _label(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    point: NDArray[np.float64],
    runs: NDArray[np.int64],
    gaps: NDArray[np.int64],
) -> int:  # pragma: no cover
    """Label a simplex point by the fixed-point rule for func in its native space.

    ``func`` acts on its native space, so the point is mapped into that space and its
    image mapped back to the simplex before the most-decreasing coordinate is taken.

    Parameters
    ----------
    func : The map whose fixed point is sought, acting on the space that ``runs`` selects.
    point : The point on the d-simplex to label.
    runs : The run shape selecting the space: empty is the hypercube, a single run is the
        simplex, and two or more runs is the simplotope those runs describe.
    gaps : The cumulative run boundaries, ``np.insert(runs.cumsum(), 0, 0)``; used only
        for the simplotope space.

    Returns
    -------
    label : The labeled coordinate, in ``[0, point.size)``.
    """
    if runs.size == 0:
        trans = hypercube_to_simplex(func(simplex_to_hypercube(point)))
    elif runs.size == 1:
        trans = func(point)
    else:
        tope = func(simplex_to_simplotope(point, runs, gaps))
        trans = simplotope_to_simplex(tope, runs, gaps)
    return int(np.argmin((point == 0) - point + trans))


@jit(
    float64[::1](_SIMPLEX_MAP, float64[:], int64[::1], int64[::1], int64),
    nogil=True,
    error_model="numpy",
)
def _pivot(  # noqa: PLR0912, PLR0915
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    init: NDArray[np.float64],
    runs: NDArray[np.int64],
    gaps: NDArray[np.int64],
    disc: int,
) -> NDArray[np.float64]:  # pragma: no cover
    """Find the approximate center of a fully labeled subsimplex.

    This is the shared sandwich-method pivot. Because ``func`` is passed as a first-class
    function, it compiles a single time for every map rather than specializing per func.

    Parameters
    ----------
    func : The map whose fixed point is sought, acting on the space that ``runs`` selects.
    init : The starting point on the d-simplex.
    runs : The run shape selecting how ``func`` is applied (see ``_label``): empty is the
        hypercube, a single run the simplex, and two or more the simplotope.
    gaps : The cumulative run boundaries; used only for the simplotope space.
    disc : The discretization to use. The returned point lies in a completely labeled
        subsimplex whose vertices are within ``1 / disc`` of each other.

    Returns
    -------
    ret : A point on the d-simplex approximating a fixed point of ``func``.
    """
    if disc < 2:  # noqa: PLR2004
        raise ValueError("discretization must be at least two")
    elif not np.all(init >= 0) or abs(1 - init.sum()) > _SIMPLEX_TOL:
        raise ValueError("must start as a valid simplex")

    dim = init.size
    # Base vertex of the subsimplex currently being used. init is normalized to
    # sum to exactly one so the discretization rounds up a valid number of slots.
    dinit = discretize_mixture(init / init.sum(), disc)
    base = np.append(dinit, 0)
    base[0] += 1
    # permutation array of [1,dim] where v0 = base,
    # v{i+1} = [..., vi_{perms[i] - 1} - 1, vi_{perms[i]} + 1, ...]
    perms = np.arange(1, dim + 1)
    # Array of labels for each vertex
    labels = np.arange(dim + 1)
    labels[dim] = _label(func, dinit / disc, runs, gaps)
    # Vertex used to label initial vertices (vertex[-1] == 0)
    label_vertex = base[:-1].copy()
    # Last index moved
    index = dim
    # Most recent created index, should be set to
    new_vertex = np.empty((dim + 1,), np.int64)

    while labels[index] < dim:
        # Find duplicate index. this is O(dim) but not a bottleneck
        current_label = labels[index]
        found = False
        for ind in range(dim + 1):
            if ind != index and labels[ind] == current_label:
                index = ind
                found = True
                break
        if not found:
            raise ValueError("labeling function was not proper (see help)")

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

        # Compute actual value of flipped vertex. Each flip is a +1/-1 pair, so
        # the sum is conserved at disc + 1 by construction; only the
        # non-negativity needs checking to catch an improper labeling.
        new_vertex[:] = base
        new_vertex[perms[:index]] += 1
        new_vertex[perms[:index] - 1] -= 1

        if not np.all(new_vertex >= 0):
            raise ValueError("vertex rotation failed, check labeling function")

        # Update label of new vertex
        if new_vertex[-1] == 2:  # noqa: PLR2004
            labels[index] = dim
        elif new_vertex[-1] == 0:
            labels[index] = np.argmax(new_vertex[:-1] - label_vertex)
        else:  # == 1
            labels[index] = _label(func, new_vertex[:-1] / disc, runs, gaps)
            if not (0 <= labels[index] < dim and new_vertex[labels[index]]):
                raise ValueError("labeling function was not proper (see help)")

    # The loop can only exit by pushing the final vertex onto layer two, so the
    # completely labeled facet is the first dim vertices; average those.
    if index != dim:
        raise ValueError("pivot exited on an unexpected vertex, check labeling function")
    current = base
    mean = current.astype(np.float64)
    for count, j in enumerate(perms[:-1], 2):
        current[j] += 1
        current[j - 1] -= 1
        mean += (current - mean) / count
    return mean[:-1] / disc


@jit(float64[::1](_SIMPLEX_MAP, float64[:], int64), nogil=True, error_model="numpy")
def labeled_subsimplex(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    init: NDArray[np.float64],
    disc: int,
) -> NDArray[np.float64]:  # pragma: no cover
    """Find the approximate center of a fully labeled subsimplex.

    This runs once at the discretization provided. It is recommended that this
    be run several times with successively finer discretization and warm
    started with the past result.

    Parameters
    ----------
    func : A continuous simplex-to-simplex map whose fixed point is sought. Each
        vertex is labeled by the coordinate that decreases most under func.
    init : The starting point on the d-simplex.
    disc : The discretization to use. The returned point lies in a completely
        labeled subsimplex whose vertices are within ``1 / disc`` of each other.

    Returns
    -------
    ret : A point on the d-simplex approximating a fixed point of ``func``.

    Notes
    -----
    This is an implementation of the sandwich method from [1]_ and [2]_.

    .. [1] Kuhn and Mackinnon 1975. Sandwich Method for Finding Fixed Points.
    .. [2] Kuhn 1968. Simplicial Approximation Of Fixed Points.
    """
    single_run = np.full(1, init.size, np.int64)
    no_gaps = np.empty(0, np.int64)
    return _pivot(func, init, single_run, no_gaps, disc)


def simplotope_fixed_point(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    init: NDArray[np.float64],
    runs: NDArray[np.int64],
    disc: int,
) -> NDArray[np.float64]:
    """Compute an approximate fixed point of a function on the simplotope.

    Each simplex should be stacked next to each other, and runs specifies the
    number of elements in each simplex.

    Parameters
    ----------
    func : A continuous function mapping from the d-simplotope (specified by
        runs) to itself.
    init : An initial guess for the fixed point. Since many may exist, the
        choice of starting point will affect the solution.
    runs : the number of elements in each simplex of the simplotope, in order.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal this much. Since this function computes a homeomorphism to
        the simplex, the distortion can be up to the number of simplices.
        Therefore if you want a true approximation discretization of `disc`,
        you should specify `disc * runs.size`.
    """
    if disc < 2:  # noqa: PLR2004
        raise ValueError("discretization must be at least two")
    elif np.any(runs < 1):
        raise ValueError("every run must contain at least one element")
    elif runs.sum() != init.size:
        raise ValueError("runs must sum to the number of elements in init")
    runs = np.ascontiguousarray(runs, dtype=np.int64)
    gaps = np.insert(runs.cumsum(), 0, 0)
    if np.any(init < 0) or np.any(
        np.abs(1 - np.add.reduceat(init, gaps[:-1])) > _SIMPLEX_TOL
    ):
        raise ValueError("each simplex of init must be non-negative and sum to one")

    approx = _pivot(func, simplotope_to_simplex(init, runs, gaps), runs, gaps, disc)
    return simplex_to_simplotope(approx, runs, gaps)


def hypercube_fixed_point(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    init: NDArray[np.float64],
    disc: int,
) -> NDArray[np.float64]:
    """Compute an approximate fixed point of a function on the unit hypercube.

    This function maps the unit hypercube to the simplex, and in so doing
    distorts the space. The worst distortion is at the tip of the hypercube [1,
    1, ..., 1] which has O(d) distortion, so the discretization will be off by
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
        areas of the hypercube.
    """
    if disc < 2:  # noqa: PLR2004
        raise ValueError("discretization must be at least two")
    elif np.any(init < 0) or np.any(init > 1):
        raise ValueError("init must lie in the unit hypercube [0, 1]")

    no_shape = np.empty(0, np.int64)
    approx = _pivot(func, hypercube_to_simplex(init), no_shape, no_shape, disc)
    return simplex_to_hypercube(approx)


def simplex_fixed_point(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    init: NDArray[np.float64],
    disc: int,
) -> NDArray[np.float64]:
    """Compute an approximate fixed point of a function on the simplex.

    Parameters
    ----------
    func : A continuous function mapping from the d-simplex to itself.
    init : An initial guess for the fixed point. Since many may exist, the
        choice of starting point will affect the solution.
    disc : The discretization to use. Fixed points will be approximated by the
        reciprocal of this much.
    """
    if disc < 2:  # noqa: PLR2004
        raise ValueError("discretization must be at least two")
    elif np.any(init < 0) or abs(1 - init.sum()) > _SIMPLEX_TOL:
        raise ValueError("init must be a valid simplex (non-negative, summing to one)")
    return labeled_subsimplex(func, init, disc)
