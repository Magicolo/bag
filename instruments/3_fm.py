from pyo import PyoObject, FM


def new(frequency: PyoObject) -> PyoObject:
    return FM(frequency, ratio=0.5)  # type: ignore
