from math import sqrt
from queue import Queue
from threading import Event, Thread
from typing import List, Optional, Tuple

from pyo import Server, Sine, hzToMidi, midiToHz, pa_get_output_devices  # type: ignore


class Audio:
    MAJOR = [0, 0, 2, 2, 4, 5, 5, 7, 7, 9, 9, 11]
    MINOR = [0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10]

    class Instrument:
        def __init__(self, scale: List[int]):
            self._play = Sine(freq=0, mul=0).out()
            # self._play = Pan(Sine(freq=0), mul=0).out()
            self._scale = scale

        def fade(self, volume: float, delta: float) -> bool:
            value = lerp(self._play.mul, volume, 10 * delta)
            self._play.mul = value
            return abs(volume - value) < 0.001

        def pan(self, pan: float, delta: float) -> bool:
            return True
            # value = lerp(self._play.pan, pan, 10 * delta)
            # self._play.pan = value
            # return abs(pan - value) < 0.001

        def shift(self, frequency: float):
            self._play.freq = note(frequency, self._scale)

    @staticmethod
    def device() -> Optional[int]:
        for name, index in zip(*pa_get_output_devices()):
            if "analog" in name.lower():
                return index

    @staticmethod
    def run(queue: Queue, abort: Event):
        _time = None
        _instruments: List[Audio.Instrument] = []

        while not abort.is_set():
            points, volume, time = queue.get()
            delta = 0 if _time is None else (time - _time) / 1000
            _time = time

            while len(_instruments) < len(points):
                _instruments.append(Audio.Instrument(Audio.MINOR))

            volume = sqrt(clamp(volume / (len(points) + 1)))
            for instrument, (x, y) in zip(_instruments, points):
                instrument.shift(y * 500)
                instrument.pan(x, delta)
                instrument.fade(volume, delta)

            for instrument in _instruments[len(points) :]:
                instrument.fade(0, delta)

    def __init__(self):
        self._server = Server(duplex=0, nchnls=2)
        self._server.setInOutDevice(Audio.device())
        self._server.boot().start()
        self._playing: List[Sine] = []
        self._fading: List[Sine] = []
        self._stopped: List[Sine] = []
        self._time = None
        self._queue = Queue(1)
        self._abort = Event()
        self._thread = Thread(target=Audio.run, args=(self._queue, self._abort))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._abort.set()
        self._server.stop()

    def update(self, points: List[Tuple[float, float]], volume: float, time: float):
        self._queue.put((points, volume, time))


def note(frequency: float, scale: List[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)
