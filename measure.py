from time import time
from typing import Callable, Iterable


class _Block:
    def __init__(self, name: str):
        self._name = name

    def __enter__(self):
        self._then = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._now = time()
        print(f"{self._name}: {self._now - self._then:.3f}s")


def block(name: str) -> _Block:
    return _Block(name)


def loop(name: str, condition: Callable[[], bool] = lambda: True) -> Iterable[None]:
    while condition():
        then = time()
        yield None
        now = time()
        print(f"{name}: {now - then:.3f}s")
