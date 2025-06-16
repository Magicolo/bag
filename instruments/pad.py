from pyo import (
    Osc,
    PyoObject,
    Snap,
    Metro,
    TrigEnv,
    TrigXnoiseMidi,
    CosTable,
    SquareTable,
)


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    beat = Metro(0.05, 8).play()
    trigger = TrigEnv(
        beat,
        table=CosTable([(0, 0), (100, 1), (500, 0.3), (8191, 0)]),
        mul=amplitude * 4.0,
    )
    pitch = TrigXnoiseMidi(beat, dist=4, x1=20, mrange=(48, 84))
    hertz = Snap(pitch, choice=[0, 2, 3, 5, 7, 8, 10], scale=1)
    return Osc(table=SquareTable(), freq=frequency * hertz / 25, phase=0, mul=trigger)


if __name__ == "__main__":
    from pyo import Server

    server = Server().boot().start()
    instrument = new(400, 0.1)
    instrument.out()
    server.gui(locals())
