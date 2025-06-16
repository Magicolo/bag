from pyo import PyoObject, FM


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    return FM(carrier=frequency / 2, mul=amplitude * 2, ratio=[0.5, 1.3, 1.9, 2.8])  # type: ignore
