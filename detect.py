from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from threading import Thread
from typing import ClassVar, Iterable, List, Literal, Sequence, Tuple, Union
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from cv2 import circle, line, rectangle
from cv2.typing import MatLike, Scalar
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
    PoseLandmarksConnections,
)
from ultralytics import YOLO

from channel import Channel, Closed
import measure
from utility import catch
import vector
from vector import Vector

Landmarks = List[Tuple[float, float]]


class Gesture(Enum):
    NONE = 0
    CLOSED_FIST = 1
    OPEN_PALM = 2
    POINTING_UP = 3
    THUMB_DOWN = 4
    THUMB_UP = 5
    VICTORY = 6
    ILOVEYOU = 7

    @staticmethod
    def from_name(name: str):
        try:
            return Gesture[name.upper()]
        except ValueError:
            return Gesture.NONE


class Handedness(Enum):
    NONE = 0
    LEFT = -1
    RIGHT = 1

    @staticmethod
    def from_name(name: str):
        try:
            return Handedness[name.upper()]
        except ValueError:
            return Handedness.NONE


@dataclass(frozen=True)
class Landmark:
    DEFAULT: ClassVar["Landmark"]

    position: Vector
    velocity: Vector
    visibility: float
    presence: float

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

    def updated(self, landmark: NormalizedLandmark) -> "Landmark":
        x, y, z = landmark.x or 0.0, landmark.y or 0.0, landmark.z or 0.0
        return Landmark(
            position=(x, y, z),
            velocity=(x - self.x, y - self.y, z - self.z),
            visibility=landmark.visibility or 0.0,
            presence=landmark.presence or 0.0,
        )


Landmark.DEFAULT = Landmark(
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    visibility=0.0,
    presence=0.0,
)


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


@dataclass(frozen=True)
class Finger(Composite):
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
    def length(self) -> float:
        return vector.distance(*(landmark.position for landmark in self.landmarks))

    @cached_property
    def angle(self) -> float:
        return vector.angle(self.tip.position, self.dip.position, self.base.position)

    def touches(self, finger: "Finger") -> bool:
        reference = vector.distance(self.tip.position, self.dip.position)
        return vector.distance(self.tip.position, finger.tip.position) < reference


@dataclass(frozen=True)
class Hand(Composite):
    DEFAULT: ClassVar["Hand"]

    handedness: Handedness
    gesture: Gesture
    frames: int

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

    def triangle(self, hand: "Hand") -> bool:
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

    def updated(
        self,
        landmarks: Iterable[NormalizedLandmark],
        handedness: Category,
        gesture: Category,
    ) -> "Hand":
        return Hand(
            landmarks=tuple(
                landmark.updated(normalized)
                for landmark, normalized in zip(self.landmarks, landmarks)
            ),
            handedness=Handedness.from_name(handedness.category_name or ""),
            gesture=Gesture.from_name(gesture.category_name or ""),
            frames=self.frames + 1,
        )


Hand.DEFAULT = Hand(
    landmarks=tuple(Landmark.DEFAULT for _ in range(21)),
    handedness=Handedness.NONE,
    gesture=Gesture.NONE,
    frames=0,
)


class Detector:
    @staticmethod
    def cpu() -> "Detector":
        return Detector(BaseOptions.Delegate.CPU)

    @staticmethod
    def gpu() -> "Detector":
        return Detector(BaseOptions.Delegate.GPU)

    def __init__(self, device: BaseOptions.Delegate = BaseOptions.Delegate.GPU):
        self._hands = Channel[Tuple[Image, int]](), Channel[Sequence[Hand]]()
        self._poses = Channel[Tuple[Image, int]](), Channel[Sequence[Landmarks]]()
        self._threads = (
            Thread(target=catch(_hands_actor, Closed, ()), args=(*self._hands, device)),
            Thread(target=catch(_poses_actor, Closed, ()), args=(*self._poses, device)),
        )
        for thread in self._threads:
            thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hands[0].close()
        self._poses[0].close()
        for thread in self._threads:
            thread.join()

    def detect(
        self, frame: MatLike, time: int
    ) -> Tuple[Sequence[Hand], Sequence[Landmarks]]:
        with measure.block("Detect"):
            image = Image(ImageFormat.SRGB, frame)
            self._hands[0].put((image, time))
            self._poses[0].put((image, time))
            return self._hands[1].get(), self._poses[1].get()

    def draw(
        self,
        frame: MatLike,
        hands: Sequence[Hand],
        poses: Sequence[Landmarks],
    ) -> MatLike:
        with measure.block("Draw"):
            height, width, _ = frame.shape
            for hand in hands:
                low, high = vector.scale(hand.minimum, hand.maximum, 1.25)
                frame = rectangle(
                    frame,
                    (int(low[0] * width), int(low[1] * height)),
                    (int(high[0] * width), int(high[1] * height)),
                    (127, 127, 0),
                    1,
                )
                for finger in hand.fingers:
                    low, high = vector.scale(finger.minimum, finger.maximum, 1.25)
                    frame = rectangle(
                        frame,
                        (int(low[0] * width), int(low[1] * height)),
                        (int(high[0] * width), int(high[1] * height)),
                        (0, 127, 127),
                        1,
                    )

            frame = _draw_landmarks(
                frame,
                (
                    [
                        (landmark.x, landmark.y, (0, 255, 0))
                        for landmark in hand.landmarks
                    ]
                    for hand in hands
                ),
                (
                    (connection.start, connection.end, (0, 255, 0))
                    for connection in HandLandmarksConnections.HAND_CONNECTIONS
                ),
            )
            frame = _draw_landmarks(
                frame, ([(hand.x, hand.y, (255, 0, 0))] for hand in hands), []
            )
            frame = _draw_landmarks(
                frame,
                (
                    [(finger.x, finger.y, (255, 0, 0)) for finger in hand.fingers]
                    for hand in hands
                ),
                [],
            )
            frame = _draw_landmarks(
                frame,
                (
                    [(landmark[0], landmark[1], (0, 0, 255)) for landmark in pose]
                    for pose in poses
                ),
                (
                    (connection.start, connection.end, (0, 0, 255))
                    for connection in PoseLandmarksConnections.POSE_LANDMARKS
                ),
            )
            return frame

    @cached_property
    def _yolo_pose(self) -> YOLO:
        return YOLO(_model_path("yolo", "yolo11n-pose.pt")).cuda()

    @cached_property
    def _yolo_object(self) -> YOLO:
        model = YOLO(_model_path("yolo", "yolo12l.pt")).cuda()
        model.fuse()
        return model


def _hands_actor(
    receive: Channel[Tuple[Image, int]],
    send: Channel[Sequence[Hand]],
    device: BaseOptions.Delegate,
):

    def load(device: BaseOptions.Delegate) -> GestureRecognizer:
        return GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "gesture_recognizer.task"
                    ),
                    delegate=device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=4,
            )
        )

    _hands: Sequence[Hand] = ()
    with load(device) as model:
        while True:
            image, time = receive.get()
            with measure.block("Detect Hands"):
                result = model.recognize_for_video(image, time)
                defaults = (
                    Hand.DEFAULT
                    for _ in range(max(len(result.hand_landmarks) - len(_hands), 0))
                )
                _hands = tuple(
                    hand.updated(landmarks, handedness[0], gestures[0])
                    for hand, landmarks, handedness, gestures in zip(
                        (*_hands, *defaults),
                        result.hand_landmarks,
                        result.handedness,
                        result.gestures,
                    )
                )
                send.put(_hands)


def _poses_actor(
    receive: Channel[Tuple[Image, int]],
    send: Channel[Sequence[Landmarks]],
    device: BaseOptions.Delegate,
):

    def load(device: BaseOptions.Delegate) -> PoseLandmarker:
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "pose_landmarker_full.task"
                    ),
                    delegate=device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
            )
        )

    with load(device) as model:
        while True:
            image, time = receive.get()
            with measure.block("Detect Pose"):
                poses = model.detect_for_video(image, time)
                send.put(
                    tuple(
                        [(landmark.x, landmark.y) for landmark in pose]
                        for pose in poses.pose_landmarks
                    )
                )


@staticmethod
def _model_path(
    folder: Union[Literal["mediapipe"], Literal["yolo"]], name: str
) -> Path:
    return Path(__file__).parent.joinpath("models", folder, name)


@staticmethod
def _draw_landmarks(
    frame: MatLike,
    landmarks: Iterable[List[Tuple[float, float, Scalar]]],
    connections: Iterable[Tuple[int, int, Scalar]],
):
    height, width, _ = frame.shape
    for group in landmarks:
        for landmark in group:
            frame = circle(
                frame,
                (int(landmark[0] * width), int(landmark[1] * height)),
                5,
                landmark[2],
                -1,
            )
        for connection in connections:
            start = group[connection[0]]
            end = group[connection[1]]
            frame = line(
                frame,
                (int(start[0] * width), int(start[1] * height)),
                (int(end[0] * width), int(end[1] * height)),
                connection[2],
                2,
            )
    return frame
