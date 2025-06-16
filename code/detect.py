from copy import copy
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import (
    ClassVar,
    Iterable,
    Literal,
    Optional,
    Self,
    Sequence,
    Set,
    Tuple,
    Union,
)

from ultralytics import YOLO
import utility
from cv2.typing import MatLike
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmark,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureRecognizerOptions,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

from cell import Cells
import measure
from utility import run
import vector
from vector import Bound, Vector


class Gesture(IntEnum):
    NONE = 0
    CLOSED_FIST = 1
    OPEN_PALM = 2
    POINTING_UP = 3
    THUMB_DOWN = 4
    THUMB_UP = 5
    VICTORY = 6
    ILOVEYOU = 7

    @staticmethod
    def from_name(name: Optional[str]):
        if name:
            try:
                return Gesture[name.upper()]
            except ValueError:
                return Gesture.NONE
        else:
            return Gesture.NONE

    def one_shot(self) -> Sequence[float]:
        return tuple(1.0 if value == self else 0.0 for value in Gesture)


class Handedness(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2

    @staticmethod
    def from_name(name: Optional[str]):
        if name:
            try:
                return Handedness[name.upper()]
            except ValueError:
                return Handedness.NONE
        else:
            return Handedness.NONE

    def one_shot(self) -> Sequence[float]:
        return tuple(1.0 if value == self else 0.0 for value in Handedness)


@dataclass(frozen=True)
class Landmark:
    DEFAULT: ClassVar[Self]

    @staticmethod
    def new(landmark: Vector) -> "Landmark":
        return Landmark(
            position=landmark,
            velocity=(0.0, 0.0, 0.0),
        )

    position: Vector
    velocity: Vector

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]

    @property
    def speed(self) -> float:
        return vector.magnitude(self.velocity)

    def update(self, landmark: Self, delta: float) -> "Landmark":
        position = vector.lerp(self.position, landmark.position, 0.25)
        return Landmark(
            position=position,
            velocity=vector.divide(vector.subtract(position, self.position), delta),
        )

    def move(self, motion: Vector) -> "Landmark":
        return Landmark(position=vector.add(self.position, motion), velocity=motion)


Landmark.DEFAULT = Landmark(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))


@dataclass(frozen=True)
class Composite:
    landmarks: Sequence[Landmark]

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]

    @property
    def bound(self) -> Bound:
        return self.minimum, self.maximum

    @cached_property
    def position(self) -> Vector:
        return vector.mean(*(landmark.position for landmark in self.landmarks))

    @cached_property
    def velocity(self) -> Vector:
        return vector.mean(*(landmark.velocity for landmark in self.landmarks))

    @cached_property
    def speed(self) -> float:
        return vector.magnitude(self.velocity)

    @property
    def area(self) -> float:
        return vector.area(self.minimum, self.maximum)

    @property
    def volume(self) -> float:
        return vector.volume(self.minimum, self.maximum)

    @cached_property
    def minimum(self) -> Vector:
        return vector.minimum(*(landmark.position for landmark in self.landmarks))

    @cached_property
    def maximum(self) -> Vector:
        return vector.maximum(*(landmark.position for landmark in self.landmarks))

    def distance(self, other: Self, square=False) -> float:
        return sum(
            vector.distance(
                old.position, (new.x or 0.0, new.y or 0.0, new.z or 0.0), square=square
            )
            for old, new in zip(self.landmarks, other.landmarks)
        )


@dataclass(frozen=True)
class Line(Composite):
    @cached_property
    def length(self) -> float:
        return vector.distance(*(landmark.position for landmark in self.landmarks))


@dataclass(frozen=True)
class Finger(Line):
    name: str

    @property
    def tip(self) -> Landmark:
        return self.landmarks[0]

    @property
    def dip(self) -> Landmark:
        return self.landmarks[1]

    @property
    def pip(self) -> Landmark:
        return self.landmarks[2]

    @property
    def base(self) -> Landmark:
        return self.landmarks[3]

    @cached_property
    def angle(self) -> float:
        return vector.angle(self.tip.position, self.dip.position, self.base.position)

    def touches(self, finger: "Finger") -> bool:
        reference = vector.distance(self.tip.position, self.dip.position)
        return vector.distance(self.tip.position, finger.tip.position) < reference


@dataclass(frozen=True)
class Hand(Composite):
    DEFAULT: ClassVar[Self]
    LEFT: ClassVar[Self]
    RIGHT: ClassVar[Self]

    @staticmethod
    def new(
        landmarks: Iterable[Vector], handedness: Category, gesture: Category
    ) -> "Hand":
        return Hand(
            landmarks=tuple(Landmark.new(landmark) for landmark in landmarks),
            handednesses=Handedness.from_name(
                handedness.category_name or ""
            ).one_shot(),
            gestures=Gesture.from_name(gesture.category_name or "").one_shot(),
        )

    handednesses: Sequence[float] = (0.0, 0.0, 0.0)
    gestures: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frames: int = 0

    @cached_property
    def gesture(self) -> Gesture:
        return Gesture(max(enumerate(self.gestures), key=lambda pair: pair[1])[0])

    @cached_property
    def handedness(self) -> Handedness:
        return Handedness(
            max(enumerate(self.handednesses), key=lambda pair: pair[1])[0]
        )

    @cached_property
    def palm(self) -> Composite:
        return Composite(
            landmarks=(
                self.landmarks[HandLandmark.WRIST],
                self.landmarks[HandLandmark.THUMB_CMC],
                self.landmarks[HandLandmark.INDEX_FINGER_MCP],
                self.landmarks[HandLandmark.MIDDLE_FINGER_MCP],
                self.landmarks[HandLandmark.RING_FINGER_MCP],
                self.landmarks[HandLandmark.PINKY_MCP],
            )
        )

    @cached_property
    def thumb(self) -> Finger:
        return Finger(
            name="thumb",
            landmarks=(
                self.landmarks[HandLandmark.THUMB_TIP],
                self.landmarks[HandLandmark.THUMB_IP],
                self.landmarks[HandLandmark.THUMB_MCP],
                self.landmarks[HandLandmark.THUMB_CMC],
            ),
        )

    @cached_property
    def index(self) -> Finger:
        return Finger(
            name="index",
            landmarks=(
                self.landmarks[HandLandmark.INDEX_FINGER_TIP],
                self.landmarks[HandLandmark.INDEX_FINGER_DIP],
                self.landmarks[HandLandmark.INDEX_FINGER_PIP],
                self.landmarks[HandLandmark.INDEX_FINGER_MCP],
            ),
        )

    @cached_property
    def middle(self) -> Finger:
        return Finger(
            name="middle",
            landmarks=(
                self.landmarks[HandLandmark.MIDDLE_FINGER_TIP],
                self.landmarks[HandLandmark.MIDDLE_FINGER_DIP],
                self.landmarks[HandLandmark.MIDDLE_FINGER_PIP],
                self.landmarks[HandLandmark.MIDDLE_FINGER_MCP],
            ),
        )

    @cached_property
    def ring(self) -> Finger:
        return Finger(
            name="ring",
            landmarks=(
                self.landmarks[HandLandmark.RING_FINGER_TIP],
                self.landmarks[HandLandmark.RING_FINGER_DIP],
                self.landmarks[HandLandmark.RING_FINGER_PIP],
                self.landmarks[HandLandmark.RING_FINGER_MCP],
            ),
        )

    @cached_property
    def pinky(self) -> Finger:
        return Finger(
            name="pinky",
            landmarks=(
                self.landmarks[HandLandmark.PINKY_TIP],
                self.landmarks[HandLandmark.PINKY_DIP],
                self.landmarks[HandLandmark.PINKY_PIP],
                self.landmarks[HandLandmark.PINKY_MCP],
            ),
        )

    @property
    def fingers(self) -> Sequence[Finger]:
        return self.thumb, self.index, self.middle, self.ring, self.pinky

    @property
    def wrist(self) -> Landmark:
        return self.landmarks[HandLandmark.WRIST]

    @property
    def connections(self) -> Sequence[Tuple[Landmark, Landmark]]:
        return tuple(
            (self.landmarks[connection.start], self.landmarks[connection.end])
            for connection in HandLandmarksConnections.HAND_CONNECTIONS
        )

    def triangle(self, hand: Self) -> bool:
        if self == hand or self.handedness == hand.handedness:
            return False
        else:
            return (
                self.thumb.touches(hand.thumb)
                and self.index.touches(hand.index)
                and not self.thumb.touches(self.index)
                and not self.thumb.touches(hand.index)
                and not self.index.touches(self.thumb)
                and not self.index.touches(hand.thumb)
            )

    def update(self, hand: Self, delta: float) -> "Hand":
        if self.frames < 5:
            landmarks = self.landmarks
        else:
            landmarks = tuple(
                old.update(new, delta)
                for old, new in zip(self.landmarks, hand.landmarks)
            )
        return Hand(
            landmarks=landmarks,
            handednesses=utility.lerp(self.handednesses, hand.handednesses, delta),
            gestures=utility.lerp(self.gestures, hand.gestures, delta * 2.5),
            frames=self.frames + 1,
        )

    def move(self, motion: Vector) -> "Hand":
        return Hand(
            landmarks=tuple(landmark.move(motion) for landmark in self.landmarks),
            handednesses=self.handednesses,
            gestures=self.gestures,
            frames=self.frames + 1,
        )


Hand.DEFAULT = Hand(landmarks=tuple(Landmark.DEFAULT for _ in range(21)))

Hand.LEFT = Hand(
    landmarks=tuple(Landmark.DEFAULT for _ in range(21)),
    handednesses=Handedness.LEFT.one_shot(),
)

Hand.RIGHT = Hand(
    landmarks=tuple(Landmark.DEFAULT for _ in range(21)),
    handednesses=Handedness.RIGHT.one_shot(),
)


@dataclass(frozen=True)
class Pose(Composite):
    DEFAULT: ClassVar[Self]

    @staticmethod
    def new(landmarks: Iterable[Vector]) -> "Pose":
        return Pose(landmarks=tuple(map(Landmark.new, landmarks)))

    frames: int = 0

    @cached_property
    def head(self) -> Composite:
        if len(self.landmarks) == 17:
            return Composite(landmarks=self.landmarks[:5])
        else:
            return Composite(landmarks=self.landmarks[:11])

    @cached_property
    def eyes(self) -> Tuple[Composite, Composite]:
        if len(self.landmarks) == 17:
            return Composite((self.landmarks[1],)), Composite((self.landmarks[2],))
        else:
            return (
                Composite(
                    landmarks=(self.landmarks[4], self.landmarks[5], self.landmarks[6])
                ),
                Composite(
                    landmarks=(self.landmarks[1], self.landmarks[2], self.landmarks[3])
                ),
            )

    @property
    def nose(self) -> Landmark:
        return self.landmarks[0]

    @cached_property
    def mouth(self) -> Line:
        if len(self.landmarks) == 17:
            return Line(landmarks=())
        else:
            return Line(landmarks=(self.landmarks[9], self.landmarks[10]))

    @property
    def ears(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return self.landmarks[3], self.landmarks[4]
        else:
            return self.landmarks[8], self.landmarks[7]

    @cached_property
    def torso(self) -> Composite:
        return Composite(
            landmarks=(self.shoulders[0], self.shoulders[1], self.hips[1], self.hips[0])
        )

    @cached_property
    def arms(self) -> Tuple[Line, Line]:
        return (
            Line(landmarks=(self.shoulders[0], self.elbows[0], self.wrists[0])),
            Line(landmarks=(self.shoulders[1], self.elbows[1], self.wrists[1])),
        )

    @cached_property
    def palms(self) -> Tuple[Composite, Composite]:
        if len(self.landmarks) == 17:
            return (
                Composite(landmarks=(self.wrists[0],)),
                Composite(landmarks=(self.wrists[1],)),
            )
        else:
            return (
                Composite(
                    landmarks=(
                        self.wrists[0],
                        self.landmarks[18],
                        self.landmarks[20],
                        self.landmarks[22],
                    )
                ),
                Composite(
                    landmarks=(
                        self.wrists[1],
                        self.landmarks[17],
                        self.landmarks[19],
                        self.landmarks[21],
                    )
                ),
            )

    @property
    def shoulders(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[5], self.landmarks[6])
        else:
            return (self.landmarks[12], self.landmarks[11])

    @property
    def elbows(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[7], self.landmarks[8])
        else:
            return (self.landmarks[14], self.landmarks[13])

    @property
    def wrists(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[9], self.landmarks[10])
        else:
            return (self.landmarks[16], self.landmarks[15])

    @cached_property
    def legs(self) -> Tuple[Line, Line]:
        return (
            Line(landmarks=(self.hips[0], self.knees[0], self.ankles[0])),
            Line(landmarks=(self.hips[1], self.knees[1], self.ankles[1])),
        )

    @cached_property
    def feet(self) -> Tuple[Composite, Composite]:
        if len(self.landmarks) == 17:
            return (
                Composite(landmarks=(self.ankles[0],)),
                Composite(landmarks=(self.ankles[1],)),
            )
        else:
            return (
                Composite(
                    landmarks=(self.ankles[0], self.landmarks[30], self.landmarks[32])
                ),
                Composite(
                    landmarks=(self.ankles[1], self.landmarks[29], self.landmarks[31])
                ),
            )

    @property
    def hips(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[11], self.landmarks[12])
        else:
            return (self.landmarks[24], self.landmarks[23])

    @property
    def knees(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[13], self.landmarks[14])
        else:
            return (self.landmarks[26], self.landmarks[25])

    @property
    def ankles(self) -> Tuple[Landmark, Landmark]:
        if len(self.landmarks) == 17:
            return (self.landmarks[15], self.landmarks[16])
        else:
            return (self.landmarks[28], self.landmarks[27])

    @property
    def connections(self) -> Sequence[Tuple[Landmark, Landmark]]:
        return (
            (self.shoulders[0], self.shoulders[1]),
            (self.hips[0], self.hips[1]),
            #########################
            (self.ears[0], self.nose),
            (self.elbows[0], self.wrists[0]),
            (self.shoulders[0], self.elbows[0]),
            (self.shoulders[0], self.hips[0]),
            (self.hips[0], self.knees[0]),
            (self.knees[0], self.ankles[0]),
            #########################
            (self.ears[1], self.nose),
            (self.elbows[1], self.wrists[1]),
            (self.shoulders[1], self.elbows[1]),
            (self.shoulders[1], self.hips[1]),
            (self.hips[1], self.knees[1]),
            (self.knees[1], self.ankles[1]),
        )

    def update(self, pose: Self, delta: float) -> "Pose":
        if self.frames < 5:
            landmarks = self.landmarks
        else:
            landmarks = tuple(
                old.update(new, delta)
                for old, new in zip(self.landmarks, pose.landmarks)
            )
        return Pose(landmarks=landmarks, frames=self.frames + 1)

    def move(self, motion: Vector) -> "Pose":
        return Pose(
            landmarks=tuple(landmark.move(motion) for landmark in self.landmarks)
        )


Pose.DEFAULT = Pose(landmarks=tuple(Landmark.DEFAULT for _ in range(33)))


@dataclass
class Player:
    pose: Pose
    hands: Tuple[Hand, Hand]


class Detector:
    def __init__(
        self,
        frame: Cells[Tuple[MatLike, int]],
        players=4,
        device: BaseOptions.Delegate = BaseOptions.Delegate.CPU,
        confidence: float = 0.5,
    ):
        self._frame = frame
        self._count = players
        self._device = device
        self._confidence = confidence
        self._hands = Cells[Sequence[Hand]]()
        self._poses = Cells[Sequence[Pose]]()
        self._players = Cells[Sequence[Player]]()
        self._threads = (
            run(self._run_hands),
            run(self._run_poses),
            run(self._run_players),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hands.close()
        self._poses.close()
        self._players.close()

    @property
    def hands(self) -> Cells[Sequence[Hand]]:
        return self._hands

    @property
    def poses(self) -> Cells[Sequence[Pose]]:
        return self._poses

    @property
    def players(self) -> Cells[Sequence[Player]]:
        return self._players

    def _mediapipe_hands(self) -> GestureRecognizer:
        return GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "gesture_recognizer.task"
                    ),
                    delegate=self._device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                min_tracking_confidence=self._confidence,
                min_hand_detection_confidence=self._confidence,
                min_hand_presence_confidence=self._confidence,
                num_hands=self._count * 2,
            )
        )

    def _mediapipe_poses(self) -> PoseLandmarker:
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "pose_landmarker_full.task"
                    ),
                    delegate=self._device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                min_pose_detection_confidence=self._confidence,
                min_pose_presence_confidence=self._confidence,
                min_tracking_confidence=self._confidence,
                num_poses=self._count,
            )
        )

    def _yolo_pose(self) -> YOLO:
        return YOLO(_model_path("yolo", "yolo11x-pose.pt")).cuda()

    def _run_hands(self):
        with self._mediapipe_hands() as _model, self._hands as _hands, self._frame.spawn() as _frame:
            for frame, time in _frame.pops():
                image = Image(ImageFormat.SRGB, frame)
                with measure.block("Hands"):
                    result = _model.recognize_for_video(image, time)
                    _hands.set(
                        tuple(
                            Hand.new(
                                (
                                    (
                                        landmark.x or 0.0,
                                        landmark.y or 0.0,
                                        landmark.z or 0.0,
                                    )
                                    for landmark in landmarks
                                ),
                                handedness[0],
                                gestures[0],
                            )
                            for landmarks, handedness, gestures in zip(
                                result.hand_landmarks,
                                result.handedness,
                                result.gestures,
                            )
                        )
                    )

    def _run_poses(self):
        _model = self._yolo_pose()
        with self._poses as _poses, self._frame.spawn() as _frame:
            for frame, _ in _frame.pops():
                with measure.block("Poses"):
                    _poses.set(
                        tuple(
                            Pose.new(
                                (
                                    (float(landmark[0]), float(landmark[1]), 0.0)
                                    for landmark in landmarks
                                ),
                            )
                            for result in _model.predict(
                                frame, stream=True, conf=self._confidence, verbose=False
                            )
                            if result.boxes
                            and result.keypoints
                            and result.keypoints.has_visible
                            for landmarks in result.keypoints.xyn
                        )
                    )

    def _run_players(self):
        _players = tuple(
            Player(Pose.DEFAULT, (Hand.LEFT, Hand.RIGHT)) for _ in range(self._count)
        )
        _now = None
        _then = None

        with self._players as _send, self._hands.spawn() as _hands, self._poses.spawn() as _poses:
            for hands, poses in zip(_hands.pops(), _poses.pops()):
                with measure.block("Players"):
                    _now = perf_counter()
                    delta = 0.0 if _then is None else _now - _then
                    _then = _now

                    hand_indices: Tuple[Set[int], Set[Tuple[int, int]]] = set(), set()
                    hand_distances = sorted(
                        (
                            (p, o, n, old.distance(new, square=True))
                            for p, player in enumerate(_players)
                            for o, old in enumerate(player.hands)
                            for n, new in enumerate(hands)
                            if new.handedness == old.handedness
                        ),
                        key=lambda pair: (pair[3], pair[0]),
                    )
                    for p, o, n, _ in hand_distances:
                        if n in hand_indices[0]:
                            continue
                        else:
                            hand_indices[0].add(n)
                            hand_indices[1].add((p, o))

                        player = _players[p]
                        old = player.hands[o]
                        new = hands[n]
                        if o == 0:
                            player.hands = (old.update(new, delta), player.hands[1])
                        else:
                            player.hands = (player.hands[0], old.update(new, delta))

                    pose_indices: Tuple[Set[int], Set[int]] = set(), set()
                    pose_distances = sorted(
                        (
                            (p, n, player.pose.distance(pose, square=True))
                            for p, player in enumerate(_players)
                            for n, pose in enumerate(poses)
                        ),
                        key=lambda pair: (pair[2], pair[0]),
                    )
                    for p, n, _ in pose_distances:
                        if n in pose_indices[0]:
                            continue
                        else:
                            pose_indices[0].add(n)
                            pose_indices[1].add(p)

                        _players[p].pose = _players[p].pose.update(poses[n], delta)

                    for p, player in enumerate(_players):
                        if p in pose_indices[1]:
                            for o, hand in enumerate(player.hands):
                                if (p, o) in hand_indices[1]:
                                    continue
                                else:
                                    hand_indices[1].add((p, o))

                                motion = vector.subtract(
                                    player.pose.wrists[o].position, hand.wrist.position
                                )
                                if o == 0:
                                    player.hands = (hand.move(motion), player.hands[1])
                                else:
                                    old = player.hands[1]
                                    player.hands = (player.hands[0], hand.move(motion))
                    _send.set(tuple(copy(player) for player in _players))


def _model_path(folder: Union[Literal["mediapipe"], Literal["yolo"]], name: str) -> str:
    return f"{Path(__file__).parent.parent.joinpath("models", folder, name)}"


# def _region(frame: MatLike, x: float, y: float, size: float = 2.0 / 3.0):
#     height, width, _ = frame.shape
#     size = int(2.0 * max(width, height) / 3.0)
#     x = (0, size) if y else (width - size, width)
#     y = (0, size) if x else (height - size, height)
#     return numpy.array(frame[y[0] : y[1], x[0] : x[1], :])
