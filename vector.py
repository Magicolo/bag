from math import acos, degrees, inf, sqrt
from typing import Tuple

Vector = Tuple[float, float, float]
ZERO = (0.0, 0.0, 0.0)
ONE = (1.0, 1.0, 1.0)
INFINITY = (inf, inf, inf)


def distance(*vectors: Vector, square: bool = False) -> float:
    last = None
    distance = 0.0
    for vector in vectors:
        distance += magnitude(subtract(vector, last or vector), square)
        last = vector
    return distance


def add(left: Vector, right: Vector) -> Vector:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def subtract(left: Vector, right: Vector) -> Vector:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def negate(vector: Vector) -> Vector:
    return -vector[0], -vector[1], -vector[2]


def divide(left: Vector, right: float) -> Vector:
    return left[0] / right, left[1] / right, left[2] / right


def multiply(left: Vector, right: float) -> Vector:
    return left[0] * right, left[1] * right, left[2] * right


def magnitude(vector: Vector, square: bool = False) -> float:
    sum = vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2
    return sum if sum == 0.0 or square else sqrt(sum)


def normalize(vector: Vector) -> Vector:
    magnitude = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude


def clamp(vector: Vector, low: Vector = ZERO, high: Vector = ONE) -> Vector:
    return maximum(minimum(vector, high), low)


def minimum(*vectors: Vector) -> Vector:
    minimum = INFINITY
    for x, y, z in vectors:
        minimum = min(minimum[0], x), min(minimum[1], y), min(minimum[2], z)
    return minimum


def maximum(*vectors: Vector) -> Vector:
    maximum = negate(INFINITY)
    for x, y, z in vectors:
        maximum = max(maximum[0], x), max(maximum[1], y), max(maximum[2], z)
    return maximum


def scale(minimum: Vector, maximum: Vector, scale: float) -> Tuple[Vector, Vector]:
    size = multiply(subtract(maximum, minimum), scale * 0.5)
    center = mean(minimum, maximum)
    return subtract(center, size), add(center, size)


def sum(*vectors: Vector) -> Vector:
    total = ZERO
    for vector in vectors:
        total = add(total, vector)
    return total


def mean(*vectors: Vector) -> Vector:
    count = 0
    total = ZERO
    for vector in vectors:
        count += 1
        total = add(total, vector)
    return divide(total, count) if count > 1 else total


def absolute(vector: Vector) -> Vector:
    return abs(vector[0]), abs(vector[1]), abs(vector[2])


def area(minimum: Vector, maximum: Vector) -> float:
    x, y, _ = subtract(maximum, minimum)
    return x * y


def volume(minimum: Vector, maximum: Vector) -> float:
    x, y, z = subtract(maximum, minimum)
    return x * y * z


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
