from typing import ClassVar, Self, Sequence, Tuple

from attr import dataclass
from cv2 import (
    COLOR_RGB2BGR,
    WINDOW_NORMAL,
    circle,
    cvtColor,
    destroyWindow,
    imshow,
    line,
    namedWindow,
    pollKey,
    rectangle,
    resize,
    resizeWindow,
)
from cv2.typing import MatLike, Scalar
from cell import Cells
from utility import run
from detect import Hand, Landmark, Player, Pose
import measure


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
        width=640,
        height=480,
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
        _draw = False
        _mute = False
        _last = None

        try:
            namedWindow(self._name, WINDOW_NORMAL)
            resizeWindow(self._name, self._width, self._height)

            with self._inputs as _send, self._frame.spawn() as _frame, self._hands.spawn() as _hands, self._poses.spawn() as _poses:
                for frame, _ in _frame.pops():
                    with measure.block("Window"):
                        if _draw:
                            height, width, _ = frame.shape
                            hands, poses = (_hands.pop(), _poses.pop())
                            for hand in hands:
                                frame = _draw_landmarks(
                                    frame, hand.landmarks, Hand.CONNECTIONS, (255, 0, 0)
                                )
                            for pose in poses:
                                frame = _draw_landmarks(
                                    frame, pose.landmarks, (), (255, 0, 0)
                                )
                                low, high = pose.bound
                                frame = rectangle(
                                    frame,
                                    (int(low[0] * width), int(low[1] * height)),
                                    (int(high[0] * width), int(high[1] * height)),
                                    (0, 255, 0),
                                    2,
                                )

                            frame = cvtColor(frame, COLOR_RGB2BGR, frame)
                            frame = resize(frame, (self._width, self._height), frame)
                            imshow(self._name, frame)
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


def _draw_landmarks(
    frame: MatLike,
    landmarks: Sequence[Landmark],
    connections: Sequence[Tuple[int, int]],
    color: Scalar,
    scale: int = 1,
):
    height, width, _ = frame.shape
    for landmark in landmarks:
        frame = circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            scale * 2,
            color,
            -1,
        )
    for connection in connections:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        frame = line(
            frame,
            (int(start.x * width), int(start.y * height)),
            (int(end.x * width), int(end.y * height)),
            color,
            scale,
        )
    return frame
