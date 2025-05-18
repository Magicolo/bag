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
    left: Tuple[float, float, float], right: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def dot(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def multiply(
    left: Tuple[float, float, float], right: float
) -> Tuple[float, float, float]:
    return left[0] * right, left[1] * right, left[2] * right


def magnitude(vector: Tuple[float, float, float]) -> float:
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def normalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    magnitude = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude
