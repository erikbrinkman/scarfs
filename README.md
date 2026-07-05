# Scarfs

[![pypi](https://img.shields.io/pypi/v/scarfs)](https://pypi.org/project/scarfs/)
[![build](https://github.com/erikbrinkman/scarfs/actions/workflows/build.yml/badge.svg)](https://github.com/erikbrinkman/scarfs/actions/workflows/build.yml)
[![docs](https://img.shields.io/badge/api-docs-blue)](https://erikbrinkman.github.io/scarfs/)

A library to find an approximate fixed point for a bounded vector valued
function.

## Installation

```bash
pip install scarfs
```

## Usage

Define the function you want to find a fixed point of using numba:

```python
import numpy as np
from numba import njit

@njit
def roll(simp: np.ndarray) -> np.ndarray:
    return np.roll(simp, 1)
```

For performance reasons, this function must be compiled by numba as a cfunc or
in nopython mode. A bare `@njit` is the simplest accepted form; if you annotate
an explicit signature, the argument must be a C-contiguous one-dimensional
`float64` array (`float64[::1]`), while the return may have any layout — so
`@jit(float64[::1](float64[::1]))` works, but a map whose *argument* is typed as
non-contiguous (`float64[:]`) is rejected with a numba `TypeError`. Jitclass
functions are currently not supported. The function
must also lie in a bounded space, three default spaces are provided: the
simplex, the simplotope, and the unit hypercube. If your bounded space is not
one of these, you'll need to first compute a homeomorphism between your space
and one of these. The main algorithm runs on the simplex, so you may find it
faster if you can project there directly.

Once your function is defined, simply call one of the fixed point functions
with an initial position and a discretization:

```python
from scarfs import simplex_fixed_point

sol = simplex_fixed_point(roll, np.array([1, 0, 0, 0], float), 100)
```

The result is guaranteed to be within `1 / discretization` of a true fixed
point (or a little larger for the other bounded spaces).

Note that fixed points are difficult to approximate generally, so this may run
for a very long time.

The public entry points validate the discretization and initial point and raise
a descriptive `ValueError` on bad values, but the map itself is trusted: passing
one with an incompatible signature surfaces as an arcane numba `TypeError`.

## Development

```sh
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

## Publishing

Releases are cut from the `release` GitHub Actions workflow, which bumps the
version, builds, and publishes to PyPI via trusted publishing.
