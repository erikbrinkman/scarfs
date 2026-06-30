"""Find approximate fixed points of bounded vector valued functions."""

from ._fixed_point import (
    hypercube_fixed_point,
    labeled_subsimplex,
    simplex_fixed_point,
    simplotope_fixed_point,
)

__all__ = (
    "hypercube_fixed_point",
    "labeled_subsimplex",
    "simplex_fixed_point",
    "simplotope_fixed_point",
)
