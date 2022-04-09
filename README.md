Scarfs
======

Library to find an approximate fixed point for a for a bounded vector valued
function.

Usage
-----

Define the function you want to find a fixed point of using numba:
```python
from numba import njit

@njit
def roll(simp: np.ndarray) -> np.ndarray:
    return np.roll(simp, 1)
```
For performance reasons, this function must be compiled by numba as a cfunc or
in nopython mode. Jitclass functions are currently not supported. The function
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

Also note that this library "trusts" you, so if you pass in invalid inputs, you
may get arcane errors.
