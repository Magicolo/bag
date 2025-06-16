from pyo import PyoObject, Osc, SquareTable


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    return Osc(SquareTable(), freq=frequency, mul=amplitude)  # type: ignore
