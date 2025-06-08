from threading import Condition, Lock
from typing import Generic, Iterable, List, Sequence, Tuple
from typing import Any, Optional, TypeVar

_T = TypeVar("_T")
_EMPTY: Any = object()
_CLOSE: Any = object()


class Closed(Exception):
    pass


class Cell(Generic[_T]):
    def __init__(self, value: _T = _EMPTY):
        self._value = value
        self._lock = Condition(Lock())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        with self._lock:
            self._value = _CLOSE
            self._lock.notify_all()

    def get(self) -> _T:
        return self._get(0.0)

    def gets(self) -> Iterable[_T]:
        return iter(self.get, _CLOSE)

    def try_get(self, wait: float = 0.0) -> Optional[_T]:
        value = self._get(wait if wait > 0.0 else None)
        return None if value is _EMPTY else value

    def set(self, value: _T) -> Optional[_T]:
        value, _ = self._set(value, True, True, 0.0)
        return None if value is _EMPTY else value

    def pop(self) -> _T:
        value, _ = self._set(_EMPTY, True, False, 0.0)
        return value

    def pops(self) -> Iterable[_T]:
        return iter(self.pop, _CLOSE)

    def try_pop(self, wait: float = 0.0) -> Optional[_T]:
        value, fail = self._set(_EMPTY, True, False, wait if wait > 0.0 else None)
        return None if fail or value is _EMPTY else value

    def try_pops(self, wait: float = 0.0) -> Iterable[Optional[_T]]:
        return iter(lambda: self.try_pop(wait), _CLOSE)

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
                    self._lock.wait(wait)
                    wait = None
                else:
                    self._lock.wait()

    def _set(
        self, value: _T, swap: bool, fill: bool, wait: Optional[float]
    ) -> Tuple[_T, bool]:
        with self._lock:
            while True:
                if self._value is _CLOSE:
                    raise Closed("cell is closed")
                elif fill and self._value is _EMPTY:
                    self._value = value
                    self._lock.notify_all()
                    return _EMPTY, False
                elif swap and self._value is not _EMPTY:
                    old = self._value
                    self._value = value
                    self._lock.notify_all()
                    return old, False
                elif wait is None:
                    return _EMPTY, True
                elif wait > 0.0:
                    self._lock.wait(wait)
                    wait = None
                else:
                    self._lock.wait()


class Cells(Generic[_T]):
    def __init__(self):
        self._lock = Lock()
        self._cells: List[Cell[_T]] = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        with self._lock:
            for cell in self._cells:
                cell.close()

    def spawn(self, value: _T = _EMPTY) -> Cell[_T]:
        with self._lock:
            cell = Cell[_T](value)
            self._cells.append(cell)
            return cell

    def get(self) -> Sequence[_T]:
        with self._lock:
            return tuple(cell.get() for cell in self._cells)

    def try_get(self, wait: float = 0.0) -> Sequence[Optional[_T]]:
        with self._lock:
            return tuple(cell.try_get(wait) for cell in self._cells)

    def set(self, value: _T) -> Sequence[Optional[_T]]:
        with self._lock:
            return tuple(cell.set(value) for cell in self._cells)

    def pop(self) -> Sequence[_T]:
        with self._lock:
            return tuple(cell.pop() for cell in self._cells)

    def try_pop(self, wait: float = 0.0) -> Sequence[Optional[_T]]:
        with self._lock:
            return tuple(cell.try_pop(wait) for cell in self._cells)

    def swap(self, value: _T) -> Sequence[_T]:
        with self._lock:
            return tuple(cell.swap(value) for cell in self._cells)

    def try_swap(self, value: _T, wait: float = 0.0) -> Sequence[Optional[_T]]:
        with self._lock:
            return tuple(cell.try_swap(value, wait) for cell in self._cells)

    def fill(self, value: _T):
        with self._lock:
            for cell in self._cells:
                cell.fill(value)

    def try_fill(self, value: _T, wait: float = 0.0) -> Sequence[bool]:
        with self._lock:
            return tuple(cell.try_fill(value, wait) for cell in self._cells)
