from dataclasses import dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from runpy import run_path
from threading import Thread
from typing import Callable, ClassVar, Iterable, List, Optional, Sequence, Set, Tuple

from pyo import Server, PyoObject, Sine, Pan, SigTo, midiToHz, hzToMidi, pa_get_output_devices  # type: ignore
from channel import Channel, Closed
from detect import Gesture, Hand
import measure
from utility import catch, clamp, cut, debug
import vector

_PAD = (-1000, -1000, -1000, -1000, -1000)


class Notes(Tuple[int], Enum):
    PENTA = (0, 0, 2, 2, 4, 4, 4, 7, 7, 9, 9, 9)
    NATURAL = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10)
    HARMONIC = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11)
    MELODIC = (0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11)
    SECRET = (*_PAD, 19, 18, 15, 9, 8, 16, 20, 24, *_PAD)


# TODO: Require some gesture to start the interaction?
# TODO: Create an evaluation loop to tweak the instruments. Use Python's eval and an 'instruments' folder.
# TODO: Finger that touch must charge a note.
# TODO: Convert kick/punch impacts in cymbal/percussion-like sounds.
# TODO: Use handedness to choose the instrument.
# TODO: When 4 right hands are detected, make it special.
# TODO: When 4 left hands are detected, make it special.


@dataclass(frozen=True)
class Factory:
    DEFAULT: ClassVar["Factory"]

    new: Callable[[PyoObject, PyoObject], PyoObject]
    name: str
    stamp: float


Factory.DEFAULT = Factory(
    new=lambda frequency: Sine(freq=frequency),  # type: ignore
    name="default",
    stamp=0.0,
)


@dataclass(frozen=True)
class Sound:
    frequency: float
    amplitude: float
    pan: float
    notes: Sequence[int]
    glide: float


class Instrument:
    def __init__(self, factory: Factory):
        self._factory = factory
        self._frequency = SigTo(0.0)
        self._amplitude = SigTo(0.0)
        self._pan = SigTo(0.5)
        self._base = self._factory.new(self._frequency, self._amplitude)
        # self._synthesizer = Freeverb(
        #     Compress(
        #         Pan(self._base, outs=8, pan=self._pan),  # type: ignore
        #         risetime=0.001,  # type: ignore
        #         knee=1.0,  # type: ignore
        #         lookahead=0.0,  # type: ignore
        #         thresh=-6.0,  # type: ignore
        #         ratio=10.0,  # type: ignore
        #     ),
        #     size=0.9,
        #     bal=0.1,
        # ).out()
        self._synthesizer = Pan(self._base, outs=2, pan=self._pan).out()  # type: ignore

    def stop(self):
        self._synthesizer.stop()

    def fade(self, volume: float):
        self._amplitude.value = volume

    def pan(self, pan: float):
        self._pan.value = pan

    def shift(self, frequency: float):
        self._frequency.value = frequency

    def glide(self, glide: float):
        self._frequency.time = glide


_Message = Tuple[Sequence[Hand], bool, bool]


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

    def send(self, hands: Sequence[Hand], mute: bool, reset: bool):
        self._channel.put((hands, mute, reset))


def _sound(
    x: float,
    y: float,
    speed: float,
    scale: float,
    index: int,
    glide: bool,
    notes: Sequence[int],
) -> Sound:
    floor = 0.0 if scale <= 0.0 else scale / 10.0
    range = 1000.0 if scale <= 0.0 else 50.0 / scale
    return Sound(
        frequency=clamp(1 - y) * range * (index % 5 + 1) + 50.0,
        amplitude=clamp(cut(speed, floor) * 100.0),
        pan=clamp(x),
        notes=notes,
        glide=0.25 if glide else 0.025,
    )


def _secret(hand: Hand, hands: Sequence[Hand], skip: Set[Hand]) -> Optional[Sound]:
    for other in hands:
        if other in skip:
            continue
        elif hand.triangle(other):
            skip.add(hand)
            skip.add(other)
            position = vector.mean(hand.position, other.position)
            speed = hand.speed + other.speed / 2.0
            index = int((hand.x + other.x / 2.0) * len(Notes.SECRET))
            note = Notes.SECRET[index % len(Notes.SECRET)]
            return _sound(position[0], position[1], speed, 0.0, 0, False, (note,))


def _sounds(hands: Sequence[Hand]) -> Iterable[Sound]:
    skip = set()
    for hand in hands:
        if hand in skip or hand.frames < 2:
            continue

        secret = _secret(hand, hands, skip)
        if secret:
            yield secret
            continue

        for index, finger in enumerate(hand.fingers):
            yield _sound(
                finger.tip.x,
                finger.tip.y,
                finger.tip.speed,
                finger.length,
                index,
                hand.gesture == Gesture.ILOVEYOU,
                Notes.NATURAL,
            )


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

    _server = Server(audio="pulse", sr=48000, duplex=0, ichnls=0, nchnls=2)
    _instruments: List[Instrument] = []
    _factories = tuple(factories())

    try:
        _server.setInOutDevice(_device("digital", "usb audio", "analog"))
        _server.boot().start()
        while True:
            hands, mute, reset = channel.get()

            with measure.block("Audio"):
                if reset:
                    _factories = tuple(factories())
                    for instrument in _instruments:
                        instrument.stop()
                    _instruments.clear()
                elif mute:
                    for instrument in _instruments:
                        instrument.fade(0)
                else:
                    sounds = tuple(_sounds(hands))
                    while len(_instruments) < len(sounds):
                        index = len(_instruments)
                        factory = _factories[(index // 5) % len(_factories)]
                        _instruments.append(Instrument(factory))

                    attenuate = sqrt(clamp(1 / (len(sounds) + 1))) / 100
                    for instrument, sound in zip(_instruments, sounds):
                        frequency = _note(sound.frequency, sound.notes)
                        instrument.glide(sound.glide)
                        instrument.shift(frequency)
                        instrument.pan(sound.pan)
                        instrument.fade(sound.amplitude * attenuate)

                    for instrument in _instruments[len(sounds) :]:
                        instrument.fade(0)
    finally:
        _server.stop()


def _device(*patterns: str) -> Optional[int]:
    devices = debug(pa_get_output_devices(), "=> Available Audio Devices: ")
    for pattern in patterns:
        for name, index in zip(*devices):
            if pattern.lower() in name.lower():
                return debug(index, f"=> Using Audio Device: {name}[", "]")


def _note(frequency: float, scale: Sequence[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)
