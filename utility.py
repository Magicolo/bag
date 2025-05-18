def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)
