from threading import Thread
from typing import Callable, Type, TypeVar, overload
from cell import Closed

_T = TypeVar("_T")
_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")


def debug(value: _T, pre: str = "", post: str = "", wrap: str = "") -> _T:
    print(f"{wrap}{pre}{value}{post}{wrap}")
    return value


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def cut(value: float, floor: float, default: float = 0):
    return default if value < floor else value


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)


@overload
def run(actor: Callable[[_T0], _T], argument0: _T0) -> Thread:
    (...)


@overload
def run(actor: Callable[[_T0, _T1], _T], argument0: _T0, argument1: _T1) -> Thread:
    (...)


@overload
def run(
    actor: Callable[[_T0, _T1, _T2], _T], argument0: _T0, argument1: _T1, argument2: _T2
) -> Thread:
    (...)


@overload
def run(
    actor: Callable[[_T0, _T1, _T2, _T3], _T],
    argument0: _T0,
    argument1: _T1,
    argument2: _T2,
    argument3: _T3,
) -> Thread:
    (...)


@overload
def run(
    actor: Callable[[_T0, _T1, _T2, _T3, _T4], _T],
    argument0: _T0,
    argument1: _T1,
    argument2: _T2,
    argument3: _T3,
    argument4: _T4,
) -> Thread:
    (...)


@overload
def run(
    actor: Callable[[_T0, _T1, _T2, _T3, _T4, _T5], _T],
    argument0: _T0,
    argument1: _T1,
    argument2: _T2,
    argument3: _T3,
    argument4: _T4,
    argument5: _T5,
) -> Thread:
    (...)


def run(actor: Callable, *arguments, **pairs) -> Thread:
    thread = Thread(target=catch(actor, Closed, ()), args=arguments, kwargs=pairs)
    thread.start()
    return thread


def catch(function, error: Type[Exception], default):
    def run(*arguments):
        try:
            return function(*arguments)
        except error:
            return default

    return run
