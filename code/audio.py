from dataclasses import dataclass
from enum import Enum
from math import sqrt
from pathlib import Path
from random import randrange
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

from pyo import Server, PyoObject, Sine, Pan, Adsr, SigTo, TrigEnv, Freeverb, ButHP, Mix, ButBP, HannTable, Noise, Delay, midiToHz, hzToMidi  # type: ignore
from cell import Cells
from window import Inputs
from detect import Gesture, Hand, Pose
import measure
from utility import clamp, cut, lerp, run
import vector


# TODO: Require some gesture to start the interaction (an open palm?)?
# TODO: Finger that touch must charge a note.
# TODO: Convert kick/punch impacts in cymbal/percussion-like sounds.
# TODO: Use handedness to choose the instrument.


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


def cymbal() -> Factory:
    def new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
        env = Adsr(attack=0.001, decay=1.0, sustain=0.2, release=3.0, dur=4.0, mul=amp)
        env_trig = TrigEnv(trigger, table=HannTable(), dur=0.001, mul=1)
        env.play(trigger)

        noise = Noise(mul=env * 0.7)
        freqs = [4000, 6000, 8000, 10000, 12000, 14000]
        resonators = []
        for f in freqs:
            reson = ButBP(noise, freq=f, q=10, mul=0.2)
            resonators.append(reson)

        wash = ButHP(noise, freq=5000, mul=0.3)
        crash = Mix(resonators + [wash], voices=2)
        hit = Noise(mul=env_trig * 0.5)
        hit = ButHP(hit, freq=8000)  # emphasise the very high end
        return (crash + hit).out()


class Instrument:
    def __init__(self, factory: Factory):
        self._factory = factory
        self._mute = False
        self._frequency = SigTo(0.0)
        self._amplitude = SigTo(0.0, time=0.25)
        self._pan = SigTo(0.5)
        self._reverb = SigTo(1.0)
        self._delay = SigTo(0.0)
        self._base = self._factory.new(self._frequency, self._amplitude)
        self._synthesizer = Pan(Freeverb(self._base, size=0.9, bal=0.1, mul=self._reverb) + Delay(self._base, delay=[0.13, 0.19, 0.23], feedback=0.75, mul=self._delay), outs=2, pan=self._pan).out()  # type: ignore

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
        self._frequency.value = _note(sound.frequency, sound.notes)
        self._frequency.time = sound.glide
        self._amplitude.value = sound.amplitude * attenuate
        self._pan.value = sound.pan
        self._reverb.value = clamp(1.0 - sound.echo)
        self._delay.value = clamp(sound.echo) * 5
        self._mute = False

    def mute(self) -> bool:
        if self._mute:
            return True
        else:
            self._mute = True
            self._amplitude.value = 0.0
            return False

    def stop(self):
        self.mute()
        self._synthesizer.stop()


class Cymbal(Instrument):
    @staticmethod
    def _new(frequency: PyoObject, amplitude: PyoObject) -> PyoObject:
        env = Adsr(attack=0.001, decay=1.0, sustain=0.2, release=3.0, dur=4.0, mul=amp)

        noise = Noise(mul=env * 0.7)
        freqs = [4000, 6000, 8000, 10000, 12000, 14000]
        resonators = []
        for f in freqs:
            reson = ButBP(noise, freq=f, q=10, mul=0.2)
            resonators.append(reson)

        wash = ButHP(noise, freq=5000, mul=0.3)
        crash = Mix(resonators + [wash], voices=2)
        hit = Noise(mul=env_trig * 0.5)
        hit = ButHP(hit, freq=8000)  # emphasise the very high end
        return (crash + hit).out()

    def __init__(self):
        super().__init__(Factory(Cymbal._new, "cymbal", 0.0))


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
        _groups: Dict[int, Tuple[Factory, List[Instrument]]] = {}
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
                            for _, instruments in _groups.values():
                                for instrument in instruments:
                                    instrument.mute()
                        else:
                            has = set()
                            sounds = tuple(_sounds(delta, hands))
                            attenuate = sqrt(clamp(1 / (len(sounds) + 1))) / 25
                            for group, index, sound in sounds:
                                has.add(group)
                                factory, instruments = _groups.setdefault(
                                    group, (_factories[randrange(len(_factories))], [])
                                )
                                while index >= len(instruments):
                                    instruments.append(Instrument(factory))

                                instruments[index].update(sound, attenuate)

                            for group, (_, instruments) in tuple(_groups.items()):
                                if group in has:
                                    continue

                                if all(instrument.mute() for instrument in instruments):
                                    for instrument in instruments:
                                        instrument.stop()
                                    del _groups[group]
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


def _load(groups: Dict[int, Tuple[Factory, List[Instrument]]]) -> Sequence[Factory]:
    def factories() -> Iterable[Factory]:
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

    for _, instruments in groups.values():
        for instrument in instruments:
            instrument.stop()
        instruments.clear()
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
    notes: Sequence[int],
) -> Sound:
    floor = scale * delta * 150.0
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
    group: int, delta: float, hand: Hand, hands: Sequence[Hand], skip: Set[Hand]
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

        secret = _secret(group, delta, hand, hands, skip)
        if secret:
            yield group, 0, secret
            continue

        for index, finger in enumerate(hand.fingers):
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
                Notes.NATURAL,
            )


def _note(frequency: float, scale: Sequence[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)
