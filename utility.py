from math import sqrt
from typing import Tuple, TypeVar

_T = TypeVar("_T")


def debug(value: _T) -> _T:
    print(value)
    return value


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def cut(value: float, floor: float, default: float = 0):
    return default if value < floor else value


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)


def distance(
    source: Tuple[float, float, float], target: Tuple[float, float, float]
) -> float:
    return magnitude(subtract(source, target))


def subtract(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def multiply(a: Tuple[float, float, float], b: float) -> Tuple[float, float, float]:
    return a[0] * b, a[1] * b, a[2] * b


def magnitude(vector: Tuple[float, float, float]) -> float:
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def normalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    magnitude = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude
