from collections.abc import Callable

class BenchmarkFixture:
    def __call__[**P, R](
        self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R: ...
    def pedantic[R](
        self,
        target: Callable[..., R],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
        setup: Callable[[], tuple[tuple[object, ...], dict[str, object]]] | None = ...,
        teardown: Callable[[object], object] | None = ...,
        rounds: int = ...,
        warmup_rounds: int = ...,
        iterations: int = ...,
    ) -> R: ...
