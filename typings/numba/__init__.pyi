from collections.abc import Callable
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

class _Types:
    def Array(
        self, dtype: Argument, ndim: int, layout: str, *, readonly: bool = ...
    ) -> Argument: ...
    def FunctionType(self, sig: Type, /) -> Argument: ...

types: _Types

@overload
def jit[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def jit(
    sig: Type | None = ...,
    /,
    *,
    cache: bool = ...,
    nogil: bool = ...,
    error_model: Literal["numpy", "python"] = ...,
) -> _Decorator: ...
