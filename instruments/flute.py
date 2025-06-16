from pyo import PyoObject, PinkNoise, Mix, Sine, ButLP, Blit, ButBP


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    vibrate = Sine(freq=5, mul=5, add=0)
    breath = ButLP(PinkNoise(mul=amplitude * 8.0), freq=2000)
    form = ButBP(
        Blit(
            freq=frequency * 4.0 + vibrate,
            harms=20,
            mul=amplitude * 16.0,
        ),
        freq=frequency * 8.0,
        q=4,
    )
    return Mix([form, breath * 0.5], voices=2)


if __name__ == "__main__":
    from pyo import Server

    server = Server().boot().start()
    instrument = new(frequency=400, amplitude=0.1)
    instrument.out()
    server.gui(locals())
