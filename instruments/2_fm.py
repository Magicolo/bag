from pyo import PyoObject, FM


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    return FM(carrier=frequency / 4, mul=amplitude, ratio=[0.5, 2.0])  # type: ignore
