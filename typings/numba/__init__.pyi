from collections.abc import Callable, Iterable
from typing import Literal, Protocol, overload

class _Decorator(Protocol):
    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...

class Type:
    pass

class Argument(Type):
    def __call__(self, *_: Argument) -> Type: ...

class Scalar(Argument):
    def __getitem__(self, val: slice | tuple[slice, ...]) -> Argument: ...

float64: Scalar
int64: Scalar
uint64: Scalar
uint8: Scalar
void: Argument

class _Types:
    def Array(
        self, dtype: Argument, ndim: int, layout: str, *, readonly: bool = ...
    ) -> Argument: ...

types: _Types

@overload
def njit[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def njit(
    sig: Type | None = ...,
    /,
    *,
    cache: bool = ...,
    parallel: bool = ...,
    nogil: bool = ...,
    fastmath: bool = ...,
    error_model: Literal["numpy", "python"] = ...,
) -> _Decorator: ...
def jit(
    sig: Type,
    /,
    *,
    cache: bool = ...,
    parallel: bool = ...,
    nogil: bool = ...,
    fastmath: bool = ...,
    error_model: Literal["numpy", "python"] = ...,
) -> _Decorator: ...
def cfunc(sig: Type, /, *, cache: bool = ..., nopython: bool = ...) -> _Decorator: ...
def prange(start: int) -> Iterable[int]: ...
