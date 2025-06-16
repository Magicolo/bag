from dataclasses import dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from runpy import run_path
from time import perf_counter
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pyo import Server, PyoObject, Sine, Pan, SigTo, Freeverb, Delay, midiToHz, hzToMidi  # type: ignore
from cell import Cells
from window import Inputs
from detect import Gesture, Hand, Pose
import measure
from utility import clamp, cut, debug, lerp, run
import vector
from pyaudio import PyAudio


# TODO: Require some gesture to start the interaction (an open palm?)?
# TODO: Finger that touch must charge a note.
# TODO: Convert kick/punch impacts in cymbal/percussion-like sounds.
# TODO: Use handedness to choose the instrument.


_CHANNELS = 8
_PAD = (-1000, -1000, -1000, -1000, -1000)


class Notes(Tuple[int], Enum):
    PENTA = (0, 0, 2, 2, 4, 4, 4, 7, 7, 9, 9, 9)
    NATURAL = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10)
    HARMONIC = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11)
    MELODIC = (0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11)
    SECRET = (*_PAD, 19, 18, 15, 9, 8, 16, 20, 24, *_PAD)


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
    echo: float


class Instrument:
    def __init__(self, factory: Factory, channels: int):
        self._factory = factory
        self._frequency = SigTo(0.0)
        self._amplitude = SigTo(0.0, time=0.25)
        self._pan = SigTo(0.5)
        self._reverb = SigTo(1.0)
        self._delay = SigTo(0.0)
        self._base = self._factory.new(self._frequency, self._amplitude)
        self._synthesizer = Pan(Freeverb(self._base, size=0.9, bal=0.1, mul=self._reverb) + Delay(self._base, delay=[0.13, 0.19, 0.23], feedback=0.75, mul=self._delay), outs=channels, pan=self._pan).out()  # type: ignore

    @property
    def amplitude(self):
        return self._amplitude.value

    @property
    def frequency(self):
        return self._frequency.value

    @property
    def pan(self):
        return self._pan.value

    @property
    def reverb(self):
        return self._reverb.value

    @property
    def delay(self):
        return self._delay.value

    def stop(self):
        self._synthesizer.stop()

    def fade(self, volume: float):
        self._amplitude.value = volume

    def spatialize(self, pan: float):
        self._pan.value = pan

    def shift(self, frequency: float):
        self._frequency.value = frequency

    def glide(self, glide: float):
        self._frequency.time = glide

    def echo(self, echo: float):
        self._reverb.value = clamp(1.0 - echo)
        self._delay.value = clamp(echo) * 5


class Audio:
    """
    Amplifier Layout:

    1. Center Front
    2. Center Sub
    3. Left Side
    4. Right Side

    5. Left Rear
    6. Right Rear
    7. Left Front
    8. Right Front

    POWER
    """

    def __init__(
        self,
        hands: Cells[Sequence[Hand]],
        poses: Cells[Sequence[Pose]],
        inputs: Cells[Inputs],
    ):
        self._hands = hands
        self._poses = poses
        self._inputs = inputs
        self._thread = run(self._run)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _run(self):
        _server = _initialize()
        _instruments: List[Instrument] = []
        _factories = tuple(_load())
        _old = tuple()
        _now = None
        _then = None

        try:
            with self._hands.spawn() as _hands, self._inputs.spawn() as _inputs:
                for hands, inputs in zip(_hands.pops(), _inputs.gets()):
                    _now = perf_counter()
                    delta = 0.0 if _then is None else _now - _then
                    _then = _now

                    add = tuple(Hand.DEFAULT for _ in range(len(hands) - len(_old)))
                    hands = tuple(
                        old.update(new, delta) for old, new in zip((*_old, *add), hands)
                    )
                    _old = hands

                    with measure.block("Audio"):
                        totals = (
                            sum(instrument.amplitude for instrument in _instruments),
                            sum(map(float, _server.getCurrentAmp())),
                        )
                        if totals[0] > 0.0 and totals[1] <= 0.0:
                            print("=> Restarting Server")
                            _server = _initialize(_server)
                            _factories = _reset(_instruments)

                        if inputs.reset:
                            print("=> Resetting Instruments")
                            _factories = _reset(_instruments)
                        elif inputs.mute:
                            for instrument in _instruments:
                                instrument.fade(0)
                        else:
                            sounds = tuple(_sounds(delta, hands))
                            while len(_instruments) < len(sounds):
                                index = len(_instruments)
                                factory = _factories[(index // 5) % len(_factories)]
                                _instruments.append(
                                    Instrument(factory, _server.getNchnls())
                                )

                            attenuate = sqrt(clamp(1 / (len(sounds) + 1))) / 25
                            for instrument, sound in zip(_instruments, sounds):
                                frequency = _note(sound.frequency, sound.notes)
                                instrument.glide(sound.glide)
                                instrument.echo(sound.echo)
                                instrument.shift(frequency)
                                instrument.spatialize(sound.pan)
                                instrument.fade(sound.amplitude * attenuate)

                            for instrument in _instruments[len(sounds) :]:
                                instrument.fade(0)
        finally:
            _server.stop()
            _server.shutdown()


def _initialize(server: Optional[Server] = None) -> Server:
    if server:
        server.stop()
        server.shutdown()

    return Server().boot().start()


def _reset(instruments: List[Instrument]) -> Sequence[Factory]:
    for instrument in instruments:
        instrument.stop()
    instruments.clear()
    return tuple(_load())


def _load() -> Iterable[Factory]:
    for file in sorted(
        Path(__file__).parent.parent.joinpath("instruments").iterdir(),
        key=lambda file: file.stem,
    ):
        if file.is_file() and file.suffix == ".py":
            name = file.stem
            stamp = file.stat().st_mtime
            new = run_path(f"{file}").get("new", None)
            if new:
                yield Factory(new=new, name=name, stamp=stamp)


def _sound(
    x: float,
    y: float,
    speed: float,
    scale: float,
    index: int,
    glide: float,
    echo: float,
    delta: float,
    notes: Sequence[int],
) -> Sound:
    floor = scale * delta * 150.0
    range = 1000.0 if scale <= 0.0 else 50.0 / scale
    return Sound(
        frequency=clamp(1 - y) * range * (index % 5 + 1) + 50.0,
        amplitude=clamp(cut(speed, floor) * 100.0),
        pan=clamp(1 - x),
        notes=notes,
        glide=lerp(0.025, 0.25, glide),
        echo=echo,
    )


def _secret(
    delta: float, hand: Hand, hands: Sequence[Hand], skip: Set[Hand]
) -> Optional[Sound]:
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
            return _sound(
                position[0], position[1], speed, 0.0, 0, False, False, delta, (note,)
            )


def _sounds(delta: float, hands: Sequence[Hand]) -> Iterable[Sound]:
    skip = set()
    for hand in hands:
        if hand in skip:
            continue

        secret = _secret(delta, hand, hands, skip)
        if secret:
            yield secret
            continue

        for index, finger in enumerate(hand.fingers):
            speed = finger.tip.speed * clamp(1.0 - hand.gestures[Gesture.CLOSED_FIST])

            yield _sound(
                finger.tip.x,
                finger.tip.y,
                speed,
                finger.length,
                index,
                hand.gestures[Gesture.ILOVEYOU],
                hand.gestures[Gesture.CLOSED_FIST],
                delta,
                Notes.NATURAL,
            )


def _device(*patterns: str) -> Optional[Mapping[str, Any]]:
    audio = PyAudio()
    try:
        devices = tuple(
            audio.get_device_info_by_index(index)
            for index in range(audio.get_device_count())
        )
        print(
            "=> Available Audio Output Devices: ",
            *(device.get("name") for device in devices),
        )

        for pattern in patterns:
            for device in devices:
                name = f"{device.get("name", "")}"
                if pattern.lower() in name.lower():
                    return debug(device, f"=> Using Audio Device: {name}[", "]")
    finally:
        audio.terminate()


def _note(frequency: float, scale: Sequence[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)
