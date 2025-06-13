import itertools
import sys
from threading import Lock
from time import perf_counter
from typing import Callable, Iterable, TypeVar


_T = TypeVar("_T")
_LOCK = Lock()
_REGISTER = {}


class _Block:
    def __init__(self, key: str):
        self._key = key

    def __enter__(self):
        self._then = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _LOCK, _REGISTER

        self._now = perf_counter()
        with _LOCK:
            _REGISTER.setdefault(self._key, []).append(self._now - self._then)


def block(key: str) -> _Block:
    return _Block(key)


def loop(
    key: str, condition: Callable[[], bool] = lambda: True, count: int = sys.maxsize
) -> Iterable[int]:
    for index in itertools.count():
        if index < count and condition():
            with block(key):
                yield index
        else:
            break


def iterate(key: str, iterable: Iterable[_T]) -> Iterable[_T]:
    iterator = iter(iterable)
    while True:
        with block(key):
            value = next(iterator)
            yield value


def flush() -> str:
    global _LOCK, _REGISTER

    register = {}
    with _LOCK:
        for key, values in _REGISTER.items():
            if values:
                register[key] = sum(values), len(values)
                values.clear()
    return " | ".join(
        f"{key}({count}): {sum / count:.5f}s" for key, (sum, count) in register.items()
    )
