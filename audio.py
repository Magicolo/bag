from math import sqrt
from queue import Queue
from threading import Event, Thread
from typing import List, Optional, Tuple

from pyo import midiToHz  # type: ignore
from pyo import Pan, Server, Sine, hzToMidi, pa_get_output_devices

from utility import clamp, lerp  # type: ignore

NATURAL = [0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10]
HARMONIC = [0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11]
MELODIC = [0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11]


class Instrument:
    def __init__(self, scale: List[int]):
        self._play = Pan(Sine(freq=0), mul=0, spread=0.1).out()
        self._scale = scale

    def fade(self, volume: float, delta: float) -> bool:
        value = lerp(self._play.mul, volume, delta)
        self._play.mul = value
        return abs(volume - value) < 0.001

    def pan(self, pan: float, delta: float) -> bool:
        value = lerp(self._play.pan, pan, delta)
        self._play.pan = value
        return abs(pan - value) < 0.001

    def shift(self, frequency: float):
        self._play.input.freq = _note(frequency, self._scale)


class Audio:
    def __init__(self):
        self._server = Server(duplex=0, nchnls=2)
        self._server.setInOutDevice(_device())
        self._server.boot().start()
        self._playing: List[Sine] = []
        self._fading: List[Sine] = []
        self._stopped: List[Sine] = []
        self._time = None
        self._queue = Queue(1)
        self._abort = Event()
        self._thread = Thread(target=_run, args=(self._queue, self._abort))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._server.stop()
        self._abort.set()

    def update(self, data: List[Tuple[float, float]], volume: float, time: float):
        self._queue.put((data, volume, time / 1000))


def _run(queue: Queue, abort: Event):
    _time = None
    _instruments: List[Instrument] = []

    while not abort.is_set():
        data, volume, time = queue.get()
        delta = 0 if _time is None else time - _time
        _time = time

        while len(_instruments) < len(data):
            _instruments.append(Instrument(NATURAL))

        volume = sqrt(clamp(volume / (len(data) + 1)))
        for instrument, (pan, frequency) in zip(_instruments, data):
            instrument.shift(frequency)
            instrument.pan(pan, delta)
            instrument.fade(volume, delta * 5)

        for instrument in _instruments[len(data) :]:
            instrument.fade(0, delta)


def _device() -> Optional[int]:
    for name, index in zip(*pa_get_output_devices()):
        if "analog" in name.lower():
            return index


def _note(frequency: float, scale: List[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)
