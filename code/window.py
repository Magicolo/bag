from math import ceil
from time import time
from typing import ClassVar, Self, Sequence, Tuple

from attr import dataclass
from cv2 import (
    COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    INTER_NEAREST,
    WINDOW_FULLSCREEN,
    WINDOW_NORMAL,
    WND_PROP_FULLSCREEN,
    Canny,
    GaussianBlur,
    addWeighted,
    circle,
    cvtColor,
    destroyWindow,
    imshow,
    line,
    namedWindow,
    pollKey,
    resize,
    resizeWindow,
    setWindowProperty,
)
from cv2.typing import MatLike, Scalar
import numpy
from cell import Cells
import vector
from utility import run
from detect import Hand, Landmark, Player, Pose
import measure
import colorsys


@dataclass(frozen=True)
class Inputs:
    DEFAULT: ClassVar[Self]

    draw: bool
    reset: bool
    mute: bool
    exit: bool


Inputs.DEFAULT = Inputs(False, False, False, False)


class Window:
    def __init__(
        self,
        frame: Cells[Tuple[MatLike, int]],
        players: Cells[Sequence[Player]],
        hands: Cells[Sequence[Hand]],
        poses: Cells[Sequence[Pose]],
        name="La Brousse À Gigante",
        width=1920,
        height=1080,
    ):
        self._frame = frame
        self._players = players
        self._hands = hands
        self._poses = poses
        self._name = name
        self._width = width
        self._height = height
        self._inputs = Cells[Inputs]()
        self._thread = run(self._run)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._inputs.close()

    @property
    def inputs(self) -> Cells[Inputs]:
        return self._inputs

    def _run(self):
        _draw = True
        _mute = False
        _last = None
        _image = None
        _previous = None

        try:
            namedWindow(self._name, WINDOW_NORMAL)
            resizeWindow(self._name, self._width, self._height)
            setWindowProperty(self._name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)

            with self._inputs as _send, self._frame.spawn() as _frame, self._hands.spawn() as _hands, self._poses.spawn() as _poses:
                for (frame, _), hands, poses in zip(
                    _frame.pops(), _hands.gets(), _poses.gets()
                ):
                    with measure.block("Window"):
                        if _draw:
                            _image = resize(frame, (self._width, self._height), _image)
                            _image = cvtColor(_image, COLOR_BGR2GRAY, _image)
                            _image = GaussianBlur(_image, (7, 7), 1.5, _image)
                            _image = Canny(_image, 50, 150, _image)
                            _image = Canny(_image, 50, 150, _image)
                            _image = cvtColor(_image, COLOR_GRAY2BGR, _image)

                            for index, hand in enumerate(hands):
                                scale = numpy.mean(
                                    tuple(finger.length * 25 for finger in hand.fingers)
                                )
                                _image = _draw_landmarks(
                                    _image,
                                    hand.landmarks,
                                    hand.connections,
                                    _color(0.1, index / 4),
                                    scale=int(ceil(scale)),
                                )
                            for index, pose in enumerate(poses):
                                _image = _draw_landmarks(
                                    _image,
                                    pose.landmarks,
                                    pose.connections,
                                    _color(0.25, index / 3, 0.5, 0.5),
                                    2,
                                )

                            if _previous is None:
                                _previous = _image.copy()
                            else:
                                _previous = GaussianBlur(
                                    _previous, (5, 5), 2.5, _previous, 2.5
                                )
                                _previous = _pixelate(_previous, (160, 90))
                                _previous = addWeighted(
                                    _image, 1.0, _previous, 0.95, 0.0
                                )
                                _image = addWeighted(
                                    _image, 1.0, _previous, 0.5, 0.0, _image
                                )
                            imshow(self._name, _image)
                        key = pollKey()
                        exit = False
                        reset = False
                        if key != _last:
                            if key == ord("d"):
                                _draw = not _draw
                            elif key == ord("r"):
                                reset = True
                            elif key == ord("m"):
                                _mute = not _mute
                            elif key in (ord("q"), 27):
                                exit = True
                        _last = key
                        _send.set(Inputs(_draw, reset, _mute, exit))
        finally:
            destroyWindow(self._name)


def _color(speed: float, offset: float, saturation=1.0, brightness=1.0) -> Scalar:
    color = colorsys.hsv_to_rgb(time() * speed + offset, saturation, brightness)
    return (
        int(color[0] * 255),
        int(color[1] * 255),
        int(color[2] * 255),
    )


def _pixelate(frame, blocks=(64, 36)):
    height, width, _ = frame.shape
    small = resize(frame, blocks, interpolation=INTER_NEAREST)
    return resize(small, (width, height), interpolation=INTER_NEAREST)


def _draw_landmarks(
    frame: MatLike,
    landmarks: Sequence[Landmark],
    connections: Sequence[Tuple[Landmark, Landmark]],
    color: Scalar,
    scale: int = 5,
):
    height, width, _ = frame.shape
    for landmark in landmarks:
        if vector.magnitude(landmark.position, square=True) > 0.0:
            frame = circle(
                frame,
                (int(landmark.x * width), int(landmark.y * height)),
                scale * 2,
                color,
                -1,
            )
    for start, end in connections:
        if (
            vector.magnitude(start.position, square=True) > 0.0
            and vector.magnitude(end.position, square=True) > 0.0
        ):
            frame = line(
                frame,
                (int(start.x * width), int(start.y * height)),
                (int(end.x * width), int(end.y * height)),
                color,
                scale,
            )
    return frame
