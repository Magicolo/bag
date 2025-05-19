from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from runpy import run_path
from threading import Thread
from typing import Callable, ClassVar, Iterable, List, Optional, Tuple

from pyo import SPan, Server, PyoObject, Sine, SigTo, Freeverb, Compress, Tone, midiToHz, hzToMidi, pa_get_output_devices  # type: ignore
from channel import Channel, Closed
from utility import catch, clamp

PENTA = (0, 0, 2, 2, 4, 4, 4, 7, 7, 9, 9, 9)
NATURAL = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10)
HARMONIC = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11)
MELODIC = (0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11)

# TODO: Create an evaluation loop to tweak the instruments. Use Python's eval and an 'instruments' folder.
# TODO: Finger that touch must charge a note.
# TODO: Convert kick/punch impacts in cymbal/percussion-like sounds.
# TODO: Use handedness to choose the instrument.


@dataclass(frozen=True)
class Factory:
    DEFAULT: ClassVar["Factory"]

    new: Callable[[PyoObject], PyoObject]
    name: str
    stamp: float


Factory.DEFAULT = Factory(
    new=lambda frequency: Sine(freq=frequency),  # type: ignore
    name="default",
    stamp=0,
)


@dataclass(frozen=True)
class Sound:
    frequency: float
    volume: float
    pan: Optional[float]
    filter: Optional[float]


class Instrument:
    def __init__(self, factory: Factory, scale: Tuple[int, ...]):
        self._factory = factory
        self._frequency = SigTo(0)
        self._volume = SigTo(0)
        self._pan = SigTo(0.5)
        self._filter = SigTo(5000)
        self._base = self._factory.new(self._frequency)
        self._synthesizer = Freeverb(
            SPan(Compress(Tone(self._base, freq=self._filter)), mul=self._volume, pan=self._pan),  # type: ignore
            size=0.9,
            bal=0.1,
        ).out()
        self._scale = scale

    def stop(self):
        self._synthesizer.stop()

    def fade(self, volume: float):
        self._volume.value = volume

    def pan(self, pan: float):
        self._pan.value = pan

    def shift(self, frequency: float):
        self._frequency.value = _note(frequency, self._scale)

    def filter(self, frequency: float):
        self._filter.value = frequency


_Message = Tuple[Tuple[Sound, ...], bool]


class Audio:
    def __init__(self):
        self._channel = Channel[_Message]()
        self._thread = Thread(target=catch(_actor, Closed, ()), args=(self._channel,))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._channel.close()
        self._thread.join()

    def send(self, sounds: Tuple[Sound, ...], reset: bool = False):
        self._channel.put((sounds, reset))


def _actor(channel: Channel[_Message]):
    def factories() -> Iterable[Factory]:
        for file in sorted(
            Path(__file__).parent.joinpath("instruments").iterdir(),
            key=lambda file: file.stem,
        ):
            if file.is_file() and file.suffix == ".py":
                name = file.stem
                stamp = file.stat().st_mtime
                new = run_path(f"{file}").get("new", None)
                if new:
                    yield Factory(new=new, name=name, stamp=stamp)

    _server = Server(duplex=0, nchnls=2)
    _instruments: List[Instrument] = []
    _factories = tuple(factories())

    try:
        _server.setInOutDevice(_device())
        _server.boot().start()
        while True:
            sounds, reset = channel.get()

            if reset:
                _factories = tuple(factories())
                for instrument in _instruments:
                    instrument.stop()
                _instruments.clear()

            while len(_instruments) < len(sounds):
                index = len(_instruments)
                factory = _factories[(index // 5) % len(_factories)]
                _instruments.append(Instrument(factory, NATURAL))

            attenuate = sqrt(clamp(1 / (len(sounds) + 1)))
            for index, (instrument, sound) in enumerate(zip(_instruments, sounds)):
                frequency = sound.frequency * (index % 5 + 1) + 50.0
                instrument.shift(frequency)
                instrument.filter(sound.filter or frequency + 500.0)
                instrument.pan(sound.pan or 0.5)
                instrument.fade(sound.volume * attenuate)

            for instrument in _instruments[len(sounds) :]:
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
