from typing import Type, TypeVar

_T = TypeVar("_T")


def debug(value: _T, pre: str = "", post: str = "", wrap: str = "") -> _T:
    print(f"{wrap}{pre}{value}{post}{wrap}")
    return value


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def cut(value: float, floor: float, default: float = 0):
    return default if value < floor else value


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)


def catch(function, error: Type[Exception], default):
    def run(*arguments):
        try:
            return function(*arguments)
        except error:
            return default

    return run
