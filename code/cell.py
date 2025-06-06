from threading import Condition, Lock
from typing import Generic, Tuple
from typing import Any, Optional, TypeVar

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Tuple)
_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")
_EMPTY: Any = object()
_CLOSE: Any = object()


class Closed(Exception):
    pass


class Cell(Generic[_T]):
    def __init__(self, value: _T = _EMPTY):
        self._value = value
        self._lock = Lock()
        self._wait = Condition(self._lock)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        with self._lock:
            self._value = _CLOSE
            self._wait.notify_all()

    def get(self) -> _T:
        return self._get(0.0)

    def try_get(self, wait: float = 0.0) -> Optional[_T]:
        value = self._get(wait if wait > 0.0 else None)
        return None if value is _EMPTY else value

    def set(self, value: _T) -> Optional[_T]:
        value, _ = self._set(value, True, True, 0.0)
        return None if value is _EMPTY else value

    def pop(self) -> _T:
        value, _ = self._set(_EMPTY, True, False, 0.0)
        return value

    def try_pop(self, wait: float = 0.0) -> Optional[_T]:
        value, fail = self._set(_EMPTY, True, False, wait if wait > 0.0 else None)
        return None if fail or value is _EMPTY else value

    def swap(self, value: _T) -> _T:
        value, _ = self._set(value, True, False, 0.0)
        return value

    def try_swap(self, value: _T, wait: float = 0.0) -> Optional[_T]:
        value, fail = self._set(value, True, False, wait if wait > 0.0 else None)
        return None if fail or value is _EMPTY else value

    def fill(self, value: _T):
        _, _ = self._set(value, False, True, 0.0)

    def try_fill(self, value: _T, wait: float = 0.0) -> bool:
        _, fail = self._set(value, False, True, wait if wait > 0.0 else None)
        return not fail

    def _get(self, wait: Optional[float]) -> _T:
        with self._lock:
            while True:
                if self._value is _CLOSE:
                    raise Closed("cell is closed")
                elif self._value is not _EMPTY:
                    return self._value
                elif wait is None:
                    return _EMPTY
                elif wait > 0.0:
                    self._wait.wait(wait)
                    wait = None
                else:
                    self._wait.wait()

    def _set(
        self, value: _T, swap: bool, fill: bool, wait: Optional[float]
    ) -> Tuple[_T, bool]:
        with self._lock:
            while True:
                if self._value is _CLOSE:
                    raise Closed("cell is closed")
                elif fill and self._value is _EMPTY:
                    self._value = value
                    self._wait.notify_all()
                    return _EMPTY, False
                elif swap and self._value is not _EMPTY:
                    old = self._value
                    self._value = value
                    self._wait.notify_all()
                    return old, False
                elif wait is None:
                    return _EMPTY, True
                elif wait > 0.0:
                    self._wait.wait(wait)
                    wait = None
                else:
                    self._wait.wait()
