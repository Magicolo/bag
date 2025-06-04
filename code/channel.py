from threading import Condition, Lock
from typing import Generic, Sequence, Tuple, overload
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


class Fuse(Generic[_F]):
    def __init__(self, channels: Sequence[Channel]):
        self._channels = channels

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for channel in self._channels:
            channel.__exit__(exc_type, exc_value, traceback)

    def close(self):
        for channel in self._channels:
            channel.close()

    def get(self) -> _F:
        return tuple(channel.get() for channel in self._channels)  # type: ignore

    def put(self, values: _F):
        for value, channel in zip(values, self._channels):
            channel.put(value)


@overload
def fuse(channel0: Channel[_T0]) -> Fuse[Tuple[_T0]]:
    (...)


@overload
def fuse(channel0: Channel[_T0], channel1: Channel[_T1]) -> Fuse[Tuple[_T0, _T1]]:
    (...)


@overload
def fuse(
    channel0: Channel[_T0], channel1: Channel[_T1], channel2: Channel[_T2]
) -> Fuse[Tuple[_T0, _T1, _T2]]:
    (...)


@overload
def fuse(
    channel0: Channel[_T0],
    channel1: Channel[_T1],
    channel2: Channel[_T2],
    channel3: Channel[_T3],
) -> Fuse[Tuple[_T0, _T1, _T2, _T3]]:
    (...)


@overload
def fuse(
    channel0: Channel[_T0],
    channel1: Channel[_T1],
    channel2: Channel[_T2],
    channel3: Channel[_T3],
    channel4: Channel[_T4],
) -> Fuse[Tuple[_T0, _T1, _T2, _T3, _T4]]:
    (...)


@overload
def fuse(
    channel0: Channel[_T0],
    channel1: Channel[_T1],
    channel2: Channel[_T2],
    channel3: Channel[_T3],
    channel4: Channel[_T4],
    channel5: Channel[_T5],
) -> Fuse[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]:
    (...)


def fuse(*channels: Channel, **pairs: Channel) -> Fuse[Tuple]:
    return Fuse(tuple((*channels, *pairs.values())))


class Broadcast(Generic[_T]):
    def __init__(self, *channels: Channel[_T]):
        self._channels = tuple(channels)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def channels(self) -> Sequence[Channel[_T]]:
        return self._channels

    def close(self):
        for channel in self._channels:
            channel.close()

    def put(self, value: _T):
        for channel in self._channels:
            channel.put(value)
