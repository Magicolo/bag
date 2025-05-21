from math import acos, degrees, sqrt
from typing import Tuple

Vector = Tuple[float, float, float]


def distance(source: Vector, target: Vector) -> float:
    return magnitude(subtract(source, target))


def add(left: Vector, right: Vector) -> Vector:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def subtract(left: Vector, right: Vector) -> Vector:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def divide(left: Vector, right: float) -> Vector:
    return left[0] / right, left[1] / right, left[2] / right


def multiply(left: Vector, right: float) -> Vector:
    return left[0] * right, left[1] * right, left[2] * right


def magnitude(vector: Vector) -> float:
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def normalize(vector: Vector) -> Vector:
    magnitude = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude


def sum(*vectors: Vector) -> Vector:
    total = 0, 0, 0
    for vector in vectors:
        total = add(total, vector)
    return total


def mean(*vectors: Vector) -> Vector:
    count = 0
    total = 0, 0, 0
    for vector in vectors:
        count += 1
        total = add(total, vector)
    return divide(total, count) if count > 1 else total


def angle(
    left: Vector,
    hinge: Vector,
    right: Vector,
) -> float:
    v1 = subtract(right, hinge)
    v2 = subtract(left, hinge)
    d = dot(v1, v2)
    m1 = magnitude(v1)
    m2 = magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    else:
        cos = max(-1.0, min(1.0, d / (m1 * m2)))
        return degrees(acos(cos))
