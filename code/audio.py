from dataclasses import dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from random import choice
from runpy import run_path
from time import perf_counter
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
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
from utility import clamp, cut, lerp, run
import vector


_PAD = (-1000, -1000, -1000, -1000, -1000)
_SECRET = (*_PAD, 19, 18, 15, 9, 8, 16, 20, 24, *_PAD)
_HARMONIC = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 8, 11)
_MELODIC = (0, 0, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11)


class Scales(Tuple[int], Enum):
    PENTA = (0, 0, 0, 3, 3, 5, 5, 7, 7, 7, 10, 10)
    NATURAL = (0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10)
    THIRD = (3, 3, 5, 5, 7, 8, 8, 10, 10, 12, 12, 14)
    FIFTH = (7, 8, 8, 10, 10, 12, 12, 14, 16, 16, 18, 18)
    SEVENTH = (10, 10, 12, 12, 14, 16, 16, 18, 18, 20, 21, 21)
    WIDE = (0, 2, 3, 5, 8, 10, 12, 14, 16, 18, 21, 24)


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
    glide: float
    echo: float
    notes: Optional[Sequence[int]]


class Instrument:
    def __init__(self, factory: Factory, octave: int, notes: Sequence[int]):
        self._factory = factory
        self._octave = octave
        self._notes = notes
        self._frequency = SigTo(0.0)
        self._mute = SigTo(1.0)
        # self._tremolo = SigTo(0.0)
        # self._amplitude = SigTo(
        #     0.0,
        #     time=0.25,
        #     mul=Sine(freq=self._tremolo * 25.0, add=1.0, mul=self._tremolo * 0.5),  # type: ignore
        # )
        self._amplitude = SigTo(0.0, time=0.25)  # type: ignore
        self._pan = SigTo(0.5)
        self._reverb = SigTo(1.0)
        self._delay = SigTo(0.0)
        self._base = self._factory.new(self._frequency, self._amplitude)
        self._synthesizer = Pan(
            Freeverb(self._base, size=0.9, bal=0.1, mul=self._reverb) + Delay(self._base, delay=[0.13, 0.19, 0.23], feedback=0.75, mul=self._delay),  # type: ignore
            outs=2,
            pan=self._pan,  # type: ignore
            mul=self._mute,  # type: ignore
        ).out()

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

    def update(self, sound: Sound, attenuate: float):
        self._frequency.value = _note(
            sound.frequency, self._octave, sound.notes or self._notes
        )
        self._frequency.time = sound.glide
        self._mute.value = 1.0
        self._amplitude.value = sound.amplitude * attenuate
        self._pan.value = sound.pan
        # self._tremolo.value = abs(sound.pan**2)
        self._reverb.value = clamp(1.0 - sound.echo)
        self._delay.value = clamp(sound.echo) * 5

    def mute(self, delta: float) -> bool:
        self._mute.value = lerp(self._mute.value, 0.0, delta * 5.0)
        return self._mute.value < 1e-5

    def try_stop(self, delta: float) -> bool:
        if self.mute(delta):
            self.stop()
            return True
        else:
            return False

    def stop(self):
        self._mute.value = 0.0
        self._amplitude.value = 0.0
        self._synthesizer.stop()


@dataclass(frozen=True)
class Group:
    @staticmethod
    def random(factories: Sequence[Factory]) -> "Group":
        return Group(
            factory=choice(factories),
            octave=choice((-1, 0, 1)),
            notes=choice(tuple(Scales)),
            instruments=[],
        )

    factory: Factory
    octave: int
    notes: Sequence[int]
    instruments: List[Instrument]

    def update(self, index: int, sound: Sound, attenuate: float):
        while index >= len(self.instruments):
            self.instruments.append(Instrument(self.factory, self.octave, self.notes))

        self.instruments[index].update(sound, attenuate)

    def clear(self):
        self.stop()
        self.instruments.clear()

    def stop(self):
        for instrument in self.instruments:
            instrument.stop()

    def try_stop(self, delta: float) -> bool:
        done = True
        for instrument in self.instruments:
            done &= instrument.try_stop(delta)
        return done

    def mute(self, delta: float) -> bool:
        done = True
        for instrument in self.instruments:
            done &= instrument.mute(delta)
        return done


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
        _server, _hum = _initialize()
        _groups: Dict[int, Group] = {}
        _factories = _load(_groups)
        _old = tuple()
        _now = None
        _then = None

        try:
            with self._hands.spawn() as _hands, self._inputs.spawn() as _inputs:
                for hands, inputs in zip(_hands.pops(), _inputs.gets()):
                    _now = perf_counter()
                    delta = 0.0 if _then is None else _now - _then
                    _then = _now
                    _server, _hum = _update(_server, _hum)

                    add = tuple(Hand.DEFAULT for _ in range(len(hands) - len(_old)))
                    hands = tuple(
                        old.update(new, delta) for old, new in zip((*_old, *add), hands)
                    )
                    _old = hands

                    with measure.block("Audio"):
                        if inputs.reset:
                            print("=> Resetting Instruments")
                            _factories = _load(_groups)
                        elif inputs.mute:
                            for group in _groups.values():
                                group.mute(delta)
                        else:
                            has = set()
                            sounds = tuple(_sounds(delta, hands))
                            attenuate = sqrt(clamp(1 / (len(sounds) + 1))) / 25
                            for key, index, sound in sounds:
                                has.add(key)
                                group = _groups.setdefault(
                                    key, Group.random(_factories)
                                )
                                group.update(index, sound, attenuate)

                            for key, group in tuple(_groups.items()):
                                if key in has:
                                    continue
                                elif group.try_stop(delta):
                                    del _groups[key]
        finally:
            _server.stop()
            _server.shutdown()


def _initialize() -> Tuple[Server, PyoObject]:
    return Server(buffersize=1024).boot().start(), Sine(freq=1, mul=0.001).out()  # type: ignore


def _update(server: Server, hum: PyoObject) -> Tuple[Server, PyoObject]:
    if sum(map(float, server.getCurrentAmp())) <= 0.0:
        hum.stop()
        server.stop()
        server.shutdown()
        return _initialize()
    else:
        return server, hum


def _load(groups: Dict[int, Group]) -> Sequence[Factory]:
    def factories() -> Iterable[Factory]:
        for file in sorted(
            Path(__file__).parent.parent.joinpath("instruments").iterdir(),
            key=lambda file: file.stem,
        ):
            if file.is_file() and file.suffix == ".py":
                name = file.stem
                stamp = file.stat().st_mtime
                factory = run_path(f"{file}")
                new = factory.get("new", None)
                if new:
                    yield Factory(new=new, name=name, stamp=stamp)

    for group in groups.values():
        for instrument in group.instruments:
            instrument.stop()
        group.instruments.clear()
    groups.clear()
    return tuple(factories())


def _sound(
    x: float,
    y: float,
    speed: float,
    scale: float,
    index: int,
    glide: float,
    echo: float,
    delta: float,
    notes: Optional[Sequence[int]] = None,
) -> Sound:
    floor = scale * delta * 100.0
    range = 1000.0 if scale <= 0.0 else 50.0 / scale
    return Sound(
        frequency=clamp(1.0 - y) * range * (index % 5 + 1) + 50.0,
        amplitude=clamp(cut(speed, floor) * 100.0),
        pan=clamp(1.0 - x),
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
            index = int((hand.x + other.x / 2.0) * len(_SECRET))
            note = _SECRET[index % len(_SECRET)]
            return _sound(
                position[0],
                position[1],
                speed,
                0.0,
                0,
                False,
                False,
                delta,
                (note,),
            )


def _sounds(delta: float, hands: Sequence[Hand]) -> Iterable[Tuple[int, int, Sound]]:
    skip = set()
    for group, hand in enumerate(hands):
        if hand in skip:
            continue

        secret = _secret(delta, hand, hands, skip)
        if secret:
            yield group, 0, secret
            continue

        for index, finger in enumerate(hand.fingers):
            notes = None
            if hand.gesture == Gesture.THUMB_UP:
                notes = _HARMONIC
            elif hand.gesture == Gesture.THUMB_DOWN:
                notes = _MELODIC

            speed = finger.tip.speed * clamp(1.0 - hand.gestures[Gesture.CLOSED_FIST])

            yield group, index, _sound(
                finger.tip.x,
                finger.tip.y,
                speed,
                finger.length,
                index,
                hand.gestures[Gesture.ILOVEYOU],
                hand.gestures[Gesture.CLOSED_FIST],
                delta,
                notes=notes,
            )


def _note(frequency: float, octave: int, scale: Sequence[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree] + octave * 12
    return midiToHz(note)
