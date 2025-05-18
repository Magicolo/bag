from math import sqrt
from threading import Thread
from typing import List, Optional, Tuple

from pyo import SPan, Server, SigTo, Freeverb, Osc, SquareTable, Tone, FM, midiToHz, hzToMidi, pa_get_output_devices  # type: ignore
from channel import Channel
from utility import clamp

PENTA = (0, 0, 2, 2, 4, 4, 4, 7, 7, 9, 9, 9)
NATURAL = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10)
HARMONIC = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11)
MELODIC = (0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11)


class Instrument:
    def __init__(self, index: int, scale: Tuple[int, ...]):
        self._index = index
        self._frequency = SigTo(0, time=0.025)
        self._volume = SigTo(0)
        self._pan = SigTo(0.5)
        self._filter = SigTo(5000)
        if self._index // 5 == 0:
            self._base = Osc(SquareTable(), freq=self._frequency)  # type: ignore
        else:
            self._base = FM(self._frequency, ratio=0.5)  # type: ignore
        self._synthesizer = Freeverb(
            SPan(Tone(self._base, freq=self._filter), mul=self._volume, pan=self._pan),  # type: ignore
            size=0.9,
            bal=0.1,
        ).out()
        self._scale = scale

    def fade(self, volume: float):
        self._volume.value = volume

    def pan(self, pan: float):
        self._pan.value = pan

    def shift(self, frequency: float):
        self._frequency.value = _note(frequency, self._scale)

    def filter(self, frequency: float):
        self._filter.value = frequency


class Audio:
    def __init__(self):
        self._channel = Channel[Tuple[Tuple[float, float, float], ...]]()
        self._thread = Thread(target=_actor, args=(self._channel,))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._channel.close()
        self._thread.join()

    def send(self, data: Tuple[Tuple[float, float, float], ...]):
        self._channel.put(data)


def _actor(channel: Channel[Tuple[Tuple[float, float, float], ...]]):
    _server = Server(duplex=0, nchnls=2)
    _instruments: List[Instrument] = []

    try:
        _server.setInOutDevice(_device())
        _server.boot().start()
        while True:
            data = channel.get()

            while len(_instruments) < len(data):
                _instruments.append(Instrument(len(_instruments), NATURAL))

            attenuate = sqrt(clamp(1 / (len(data) + 1)))
            for instrument, (pan, frequency, volume) in zip(_instruments, data):
                instrument.shift(frequency)
                instrument.filter(frequency + 500)
                instrument.pan(pan)
                instrument.fade(volume * attenuate)

            for instrument in _instruments[len(data) :]:
                instrument.fade(0)
    finally:
        _server.stop()


def _device() -> Optional[int]:
    for name, index in zip(*pa_get_output_devices()):
        if "analog" in name.lower():
            return index


def _note(frequency: float, scale: Tuple[int, ...]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)
