from pyo import Osc, Mix, PyoObject, SawTable, MoogLP, Sine, Chorus


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    saw1 = Osc(table=SawTable(order=12), freq=frequency - 0.3, mul=amplitude * 2.0)
    saw2 = Osc(table=SawTable(order=12), freq=frequency, mul=amplitude * 2.0)
    saw3 = Osc(table=SawTable(order=12), freq=frequency + 0.3, mul=amplitude * 2.0)
    cut = Sine(freq=0.05, mul=1000, add=1500)
    filter = MoogLP(Mix([saw1, saw2, saw3]), freq=cut, res=0.8).mix(2)
    return Chorus(filter, depth=1.5, feedback=0.2, bal=0.3)


if __name__ == "__main__":
    from pyo import Server

    server = Server().boot().start()
    instrument = new(400, 0.1)
    instrument.out()
    server.gui(locals())
