from pyo import PyoObject, SuperSaw, Sine


def new(frequency: PyoObject) -> PyoObject:
    return SuperSaw(frequency / 8, detune=Sine([0.4, 0.3], mul=0.2, add=0.5), mul=5)  # type: ignore
