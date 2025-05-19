from pyo import PyoObject, Osc, SquareTable


def new(frequency: PyoObject) -> PyoObject:
    return Osc(SquareTable(), freq=frequency)  # type: ignore
