from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Iterable, Literal, Optional, Self, Sequence, Tuple, Union
import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from cv2 import cvtColor
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
    PoseLandmarksConnections,
)
from ultralytics import YOLO

from channel import Channel
import measure
from utility import run
import vector
from vector import Bound, Vector


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
    def from_name(name: Optional[str]):
        if name:
            try:
                return Gesture[name.upper()]
            except ValueError:
                return Gesture.NONE
        else:
            return Gesture.NONE


class Handedness(Enum):
    NONE = 0
    LEFT = -1
    RIGHT = 1

    @staticmethod
    def from_name(name: Optional[str]):
        if name:
            try:
                return Handedness[name.upper()]
            except ValueError:
                return Handedness.NONE
        else:
            return Handedness.NONE


@dataclass(frozen=True)
class Landmark:
    DEFAULT: ClassVar[Self]

    @staticmethod
    def new(landmark: NormalizedLandmark) -> "Landmark":
        return Landmark(
            position=(landmark.x or 0.0, landmark.y or 0.0, landmark.z or 0.0),
            velocity=(0.0, 0.0, 0.0),
            visibility=landmark.visibility or 0.0,
            presence=landmark.presence or 0.0,
        )

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

    def update(self, landmark: Self) -> "Landmark":
        return Landmark(
            position=landmark.position,
            velocity=vector.subtract(landmark.position, self.position),
            visibility=landmark.visibility,
            presence=landmark.presence,
        )

    def move(self, motion: Vector) -> "Landmark":
        return Landmark(
            position=vector.add(self.position, motion),
            velocity=motion,
            visibility=self.visibility,
            presence=self.presence,
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
    DEFAULT: ClassVar[Self]
    CONNECTIONS: ClassVar[Sequence[Tuple[int, int]]] = tuple(
        (connection.start, connection.end)
        for connection in HandLandmarksConnections.HAND_CONNECTIONS
    )

    @staticmethod
    def new(
        landmarks: Iterable[NormalizedLandmark], handedness: Category, gesture: Category
    ) -> "Hand":
        return Hand(
            landmarks=tuple(Landmark.new(normalized) for normalized in landmarks),
            handedness=Handedness.from_name(handedness.category_name or ""),
            gesture=Gesture.from_name(gesture.category_name or ""),
        )

    handedness: Handedness
    gesture: Gesture

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

    def update(self, hand: Self) -> "Hand":
        return Hand(
            landmarks=tuple(
                old.update(new) for old, new in zip(self.landmarks, hand.landmarks)
            ),
            handedness=hand.handedness,
            gesture=hand.gesture,
        )

    def move(self, motion: Vector) -> "Hand":
        return Hand(
            landmarks=tuple(landmark.move(motion) for landmark in self.landmarks),
            handedness=self.handedness,
            gesture=self.gesture,
        )


Hand.DEFAULT = Hand(
    landmarks=tuple(Landmark.DEFAULT for _ in range(21)),
    handedness=Handedness.NONE,
    gesture=Gesture.NONE,
)


@dataclass(frozen=True)
class Pose(Composite):
    DEFAULT: ClassVar[Self]
    CONNECTIONS: ClassVar[Sequence[Tuple[int, int]]] = tuple(
        (connection.start, connection.end)
        for connection in PoseLandmarksConnections.POSE_LANDMARKS
    )

    @staticmethod
    def new(landmarks: Iterable[NormalizedLandmark]) -> "Pose":
        return Pose(landmarks=tuple(map(Landmark.new, landmarks)))

    @cached_property
    def eyes(self) -> Tuple[Composite, Composite]:
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
    def mouth(self) -> Composite:
        return Composite(landmarks=(self.landmarks[9], self.landmarks[10]))

    @property
    def ears(self) -> Tuple[Landmark, Landmark]:
        return self.landmarks[8], self.landmarks[7]

    @cached_property
    def palms(self) -> Tuple[Composite, Composite]:
        return (
            Composite(
                landmarks=(
                    self.landmarks[16],
                    self.landmarks[18],
                    self.landmarks[20],
                    self.landmarks[22],
                )
            ),
            Composite(
                landmarks=(
                    self.landmarks[15],
                    self.landmarks[17],
                    self.landmarks[19],
                    self.landmarks[21],
                )
            ),
        )

    @cached_property
    def arms(self) -> Tuple[Composite, Composite]:
        return (
            Composite(
                landmarks=(self.landmarks[12], self.landmarks[14], self.landmarks[16])
            ),
            Composite(
                landmarks=(self.landmarks[11], self.landmarks[13], self.landmarks[15])
            ),
        )

    @property
    def shoulders(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[12], self.landmarks[11])

    @property
    def hips(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[24], self.landmarks[23])

    @property
    def ankles(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[28], self.landmarks[27])

    @property
    def heels(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[30], self.landmarks[29])

    @property
    def elbows(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[14], self.landmarks[13])

    @property
    def wrists(self) -> Tuple[Landmark, Landmark]:
        return (self.landmarks[16], self.landmarks[15])

    @cached_property
    def head(self) -> Composite:
        return Composite(
            landmarks=self.landmarks[0:11],
        )

    @cached_property
    def torso(self) -> Composite:
        return Composite(
            landmarks=(
                self.landmarks[11],
                self.landmarks[12],
                self.landmarks[24],
                self.landmarks[23],
            )
        )

    @cached_property
    def legs(self) -> Tuple[Composite, Composite]:
        return (
            Composite(
                landmarks=(self.landmarks[24], self.landmarks[26], self.landmarks[28])
            ),
            Composite(
                landmarks=(self.landmarks[23], self.landmarks[25], self.landmarks[27])
            ),
        )

    @cached_property
    def feet(self) -> Tuple[Composite, Composite]:
        return (
            Composite(
                landmarks=(self.landmarks[28], self.landmarks[30], self.landmarks[32])
            ),
            Composite(
                landmarks=(self.landmarks[27], self.landmarks[29], self.landmarks[31])
            ),
        )

    def update(self, pose: Self) -> "Pose":
        return Pose(
            landmarks=tuple(
                old.update(new) for old, new in zip(self.landmarks, pose.landmarks)
            )
        )

    def move(self, motion: Vector) -> "Pose":
        return Pose(
            landmarks=tuple(landmark.move(motion) for landmark in self.landmarks)
        )


Pose.DEFAULT = Pose(landmarks=tuple(Landmark.DEFAULT for _ in range(33)))


class Detector:
    def __init__(
        self,
        device: BaseOptions.Delegate = BaseOptions.Delegate.GPU,
        confidence: float = 0.5,
        hands=4,
        poses=2,
    ):
        self._hands = Channel[Tuple[Image, int]](), Channel[Sequence[Hand]]()
        self._poses = Channel[Tuple[Image, int]](), Channel[Sequence[Pose]]()
        self._threads = (
            run(_hands_actor, *self._hands, device, confidence, hands),
            run(_poses_actor, *self._poses, device, confidence, poses),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hands[0].close()
        self._poses[0].close()
        for thread in self._threads:
            thread.join()

    def detect(
        self, frame: MatLike, time: int
    ) -> Tuple[Sequence[Hand], Sequence[Pose]]:
        with measure.block("Detect"):
            image = Image(ImageFormat.SRGB, cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._hands[0].put((image, time))
            self._poses[0].put((image, time))
            hands = self._hands[1].get()
            poses = self._poses[1].get()
            return hands, poses

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
    confidence: float,
    hands: int,
):

    def load(
        device: BaseOptions.Delegate, confidence: float, hands: int
    ) -> GestureRecognizer:
        return GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "gesture_recognizer.task"
                    ),
                    delegate=device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                min_tracking_confidence=confidence,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                num_hands=hands,
            )
        )

    # def load(device: BaseOptions.Delegate) -> HandLandmarker:
    #     return HandLandmarker.create_from_options(
    #         HandLandmarkerOptions(
    #             base_options=BaseOptions(
    #                 model_asset_path=_model_path("mediapipe", "hand_landmarker.task"),
    #                 delegate=device,
    #             ),
    #             running_mode=VisionTaskRunningMode.VIDEO,
    #             min_tracking_confidence=CONFIDENCE,
    #             min_hand_detection_confidence=CONFIDENCE,
    #             min_hand_presence_confidence=CONFIDENCE,
    #             num_hands=_HANDS,
    #         )
    #     )

    with load(device, confidence, hands) as model:
        while True:
            image, time = receive.get()
            with measure.block("Hands"):
                result = model.recognize_for_video(image, time)
                send.put(
                    tuple(
                        Hand.new(landmarks, handedness[0], gestures[0])
                        for landmarks, handedness, gestures in zip(
                            result.hand_landmarks, result.handedness, result.gestures
                        )
                    )
                )


def _poses_actor(
    receive: Channel[Tuple[Image, int]],
    send: Channel[Sequence[Pose]],
    device: BaseOptions.Delegate,
    confidence: float,
    poses: int,
):

    def load(
        device: BaseOptions.Delegate, confidence: float, poses: int
    ) -> PoseLandmarker:
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "pose_landmarker_full.task"
                    ),
                    delegate=device,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                min_pose_detection_confidence=confidence,
                min_pose_presence_confidence=confidence,
                min_tracking_confidence=confidence,
                num_poses=poses,
            )
        )

    with load(device, confidence, poses) as model:
        while True:
            image, time = receive.get()
            with measure.block("Poses"):
                result = model.detect_for_video(image, time)
                send.put(tuple(map(Pose.new, result.pose_landmarks)))


def _model_path(folder: Union[Literal["mediapipe"], Literal["yolo"]], name: str) -> str:
    return f"{Path(__file__).parent.parent.joinpath("models", folder, name)}"
