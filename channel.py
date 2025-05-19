from threading import Condition, Lock
from typing import Generic, Tuple
from typing import Any, Optional, TypeVar

_T = TypeVar("_T")
_EMPTY: Any = object()
_CLOSE: Any = object()


class Closed(Exception):
    pass


class Channel(Generic[_T]):
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

    def try_get(self, timeout: float = 0.0) -> Optional[_T]:
        while True:
            with self._lock:
                if self._value is _CLOSE:
                    raise Closed("channel is closed")
                elif self._value is _EMPTY:
                    if timeout <= 0.0:
                        return None
                    else:
                        self._wait.wait(timeout)
                        timeout = 0.0
                else:
                    value = self._value
                    self._value = _EMPTY
                    return value

    def get(self) -> _T:
        while True:
            with self._lock:
                if self._value is _CLOSE:
                    raise Closed("channel is closed")
                elif self._value is _EMPTY:
                    self._wait.wait()
                else:
                    value = self._value
                    self._value = _EMPTY
                    return value

    def put(self, value: _T) -> Optional[_T]:
        with self._lock:
            if self._value is _CLOSE:
                raise Closed("channel is closed")
            else:
                old = self._value
                self._value = value
                self._wait.notify()
                return None if old is _EMPTY else old


class Broadcast(Generic[_T]):
    def __init__(self, *channels: Channel[_T]):
        self._channels = tuple(channels)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def channels(self) -> Tuple[Channel[_T], ...]:
        return self._channels

    def close(self):
        for channel in self._channels:
            channel.close()

    def put(self, value: _T):
        for channel in self._channels:
            channel.put(value)
