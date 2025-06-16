from pyo import PyoObject, Osc, SquareTable, Change, TrigEnv, LinTable, SigTo


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    return Osc(
        SquareTable(),
        freq=frequency,
        mul=TrigEnv(
            Change(frequency),
            table=LinTable([(0, 0), (100, 1), (500, 0.2), (8191, 0)]),
            dur=0.2,
            mul=amplitude * 4.0,
        ),
    )


if __name__ == "__main__":
    from pyo import Server

    server = Server().boot().start()
    instrument = new(SigTo(400), SigTo(0.1))
    instrument.out()
    server.gui(locals())
