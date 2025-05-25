import itertools
import sys
from time import perf_counter
from typing import Callable, Iterable, TypeVar

_T = TypeVar("_T")


class _Block:
    def __init__(self, name: str):
        self._name = name

    def __enter__(self):
        self._then = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._now = perf_counter()
        print(f"{self._name}: {self._now - self._then:.3f}s")


def block(name: str) -> _Block:
    return _Block(name)


def loop(
    name: str, condition: Callable[[], bool] = lambda: True, count: int = sys.maxsize
) -> Iterable[None]:
    for index in itertools.count():
        if index < count and condition():
            then = perf_counter()
            yield None
            now = perf_counter()
            print(f"{name}({index}): {now - then:.3f}s")
        else:
            break


def iterate(name: str, iterable: Iterable[_T]) -> Iterable[_T]:
    iterator = iter(iterable)
    while True:
        then = perf_counter()
        value = next(iterator)
        now = perf_counter()
        print(f"{name}: {now - then:.3f}s")
        yield value
