from pyo import PyoObject, Noise, PinkNoise, Reson, Mix, Sine


def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
    vibrate = Sine(freq=5, mul=2.5, add=0)
    noise = Noise(mul=amplitude * 32) + PinkNoise(mul=amplitude * 16)
    bell = Mix(
        [
            Reson(noise, freq=(frequency * ratio) + vibrate * index, q=64 + index * 32)
            for index, ratio in enumerate((1.00, 2.00, 2.72, 3.99, 5.28))
        ],
        voices=2,
    )
    return bell


if __name__ == "__main__":
    from pyo import Server

    server = Server().boot().start()
    instrument = new(frequency=400, amplitude=0.1)
    instrument.out()
    server.gui(locals())
