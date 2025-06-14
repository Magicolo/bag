from time import sleep
from pyo import Sine, Pan
import audio


server = audio._initialize()
try:
    Pan(Sine(freq=Sine(freq=1, mul=100, add=500)), outs=server.getNchnls()).out()  # type: ignore
    while True:
        sleep(1)
finally:
    server.stop()
    server.shutdown()
