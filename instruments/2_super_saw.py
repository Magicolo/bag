from pyo import PyoObject, SuperSaw


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    return SuperSaw(freq=frequency / 4, mul=amplitude * 8)  # type: ignore
