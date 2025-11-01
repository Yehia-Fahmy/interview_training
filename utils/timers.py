import time
from contextlib import contextmanager
from typing import Iterator, Callable, Any


@contextmanager
def timer(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        print(f"[timer] {name}: {dur_ms:.2f} ms")


def time_function(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    start = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000.0


