from dataclasses import dataclass
from functools import cached_property
from math import acos, degrees
from posixpath import dirname, join
from queue import Queue
from threading import Thread
from typing import ClassVar, Iterable, List, Literal, Tuple, Union
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from cv2 import COLOR_BGR2RGB, circle, cvtColor, line
from cv2.typing import MatLike, Scalar
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmark,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from ultralytics import YOLO

from channel import Channel
from utility import dot, magnitude, subtract

Landmarks = List[Tuple[float, float]]


@dataclass(frozen=True)
class Landmark:
    DEFAULT: ClassVar["Landmark"]

    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
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
        return magnitude(self.velocity)

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
class Finger:
    tip: Landmark
    dip: Landmark
    base: Landmark

    @property
    def angle(self) -> float:
        v1 = subtract(self.base.position, self.dip.position)
        v2 = subtract(self.tip.position, self.dip.position)
        d = dot(v1, v2)
        m1 = magnitude(v1)
        m2 = magnitude(v2)
        if m1 == 0 or m2 == 0:
            return 0.0
        else:
            cos = max(-1.0, min(1.0, d / (m1 * m2)))
            return degrees(acos(cos))


@dataclass(frozen=True)
class Hand:
    DEFAULT: ClassVar["Hand"]

    landmarks: Tuple[Landmark, ...]
    handedness: float
    """
    Returns:
        float: -1 is left handed, 1 is right handed
    """

    @property
    def thumb(self) -> Finger:
        return Finger(
            tip=self.landmarks[HandLandmark.THUMB_TIP],
            dip=self.landmarks[HandLandmark.THUMB_IP],
            base=self.landmarks[HandLandmark.THUMB_MCP],
        )

    @property
    def index(self) -> Finger:
        return Finger(
            tip=self.landmarks[HandLandmark.INDEX_FINGER_TIP],
            dip=self.landmarks[HandLandmark.INDEX_FINGER_PIP],
            base=self.landmarks[HandLandmark.INDEX_FINGER_MCP],
        )

    @property
    def middle(self) -> Finger:
        return Finger(
            tip=self.landmarks[HandLandmark.MIDDLE_FINGER_TIP],
            dip=self.landmarks[HandLandmark.MIDDLE_FINGER_PIP],
            base=self.landmarks[HandLandmark.MIDDLE_FINGER_MCP],
        )

    @property
    def ring(self) -> Finger:
        return Finger(
            tip=self.landmarks[HandLandmark.RING_FINGER_TIP],
            dip=self.landmarks[HandLandmark.RING_FINGER_PIP],
            base=self.landmarks[HandLandmark.RING_FINGER_MCP],
        )

    @property
    def pinky(self) -> Finger:
        return Finger(
            tip=self.landmarks[HandLandmark.PINKY_TIP],
            dip=self.landmarks[HandLandmark.PINKY_PIP],
            base=self.landmarks[HandLandmark.PINKY_MCP],
        )

    @property
    def fingers(self) -> Tuple[Finger, ...]:
        return self.thumb, self.index, self.middle, self.ring, self.pinky

    @property
    def finger_tips(self) -> Tuple[Landmark, ...]:
        return (
            self.landmarks[HandLandmark.THUMB_TIP],
            self.landmarks[HandLandmark.INDEX_FINGER_TIP],
            self.landmarks[HandLandmark.MIDDLE_FINGER_TIP],
            self.landmarks[HandLandmark.RING_FINGER_TIP],
            self.landmarks[HandLandmark.PINKY_TIP],
        )

    @property
    def finger_bases(self) -> Tuple[Landmark, ...]:
        return (
            self.landmarks[HandLandmark.THUMB_MCP],
            self.landmarks[HandLandmark.INDEX_FINGER_MCP],
            self.landmarks[HandLandmark.MIDDLE_FINGER_MCP],
            self.landmarks[HandLandmark.RING_FINGER_MCP],
            self.landmarks[HandLandmark.PINKY_MCP],
        )

    @property
    def wrist(self) -> Landmark:
        return self.landmarks[HandLandmark.WRIST]

    def updated(
        self, landmarks: Iterable[NormalizedLandmark], category: Category
    ) -> "Hand":
        return Hand(
            landmarks=tuple(
                landmark.updated(normalized)
                for landmark, normalized in zip(self.landmarks, landmarks)
            ),
            handedness=(
                category.score or 0.0
                if category.index == 0
                else -(category.score or 0.0) if category.index == 1 else 0.0
            ),
        )


Hand.DEFAULT = Hand(
    landmarks=tuple(Landmark.DEFAULT for _ in range(21)),
    handedness=0.0,
)


class Detector:
    def __init__(self):
        self._hands = Channel[Tuple[Image, int]](), Channel[Tuple[Hand, ...]]()
        self._poses = Channel[Tuple[Image, int]](), Channel[List[Landmarks]]()
        self._threads = (
            Thread(target=_hands_actor, args=self._hands),
            Thread(target=_poses_actor, args=self._poses),
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
    ) -> Tuple[Tuple[Hand, ...], List[Landmarks]]:
        image = Image(ImageFormat.SRGB, cvtColor(frame, COLOR_BGR2RGB))
        self._hands[0].put((image, time))
        self._poses[0].put((image, time))
        return self._hands[1].get(), self._poses[1].get()

    def draw(
        self,
        frame: MatLike,
        hands: Tuple[Hand, ...],
        poses: List[Landmarks],
    ) -> MatLike:
        frame = _draw_landmarks(
            frame,
            (0, 255, 0),
            (
                [(landmark.x, landmark.y) for landmark in hand.landmarks]
                for hand in hands
            ),
            (
                (connection.start, connection.end)
                for connection in HandLandmarksConnections.HAND_CONNECTIONS
            ),
        )
        frame = _draw_landmarks(
            frame,
            (0, 0, 255),
            poses,
            (
                (connection.start, connection.end)
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


def _hands_actor(receive: Channel[Tuple[Image, int]], send: Channel[Tuple[Hand, ...]]):
    def load() -> HandLandmarker:
        return HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path("mediapipe", "hand_landmarker.task"),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=4,
            )
        )

    _hands: Tuple[Hand, ...] = ()
    with load() as model:
        while True:
            image, time = receive.get()
            result = model.detect_for_video(image, time)
            defaults = (
                Hand.DEFAULT
                for _ in range(max(len(result.hand_landmarks) - len(_hands), 0))
            )
            _hands = tuple(
                hand.updated(landmarks, handedness[0])
                for hand, landmarks, handedness in zip(
                    (*_hands, *defaults),
                    result.hand_landmarks,
                    result.handedness,
                )
            )
            send.put(_hands)


def _poses_actor(receive: Queue, send: Queue):
    def load() -> PoseLandmarker:
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=_model_path(
                        "mediapipe", "pose_landmarker_full.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
            )
        )

    with load() as model:
        while True:
            message = receive.get()
            if message is None:
                break

            image, time = message
            poses = model.detect_for_video(image, time)
            send.put(
                [
                    [(landmark.x, landmark.y) for landmark in pose]
                    for pose in poses.pose_landmarks
                ]
            )


def _predict_yolo_pose(model: YOLO, frame: MatLike) -> Iterable[Landmarks]:
    for results in model.predict(frame, stream=True):
        if results.keypoints:
            for pose in results.keypoints.xyn:
                yield pose.tolist()


@staticmethod
def _model_path(folder: Union[Literal["mediapipe"], Literal["yolo"]], name: str) -> str:
    return join(dirname(__file__), "models", folder, name)


@staticmethod
def _draw_landmarks(
    frame: MatLike,
    color: Scalar,
    landmarks: Iterable[Landmarks],
    connections: Iterable[Tuple[int, int]],
):
    height, width, _ = frame.shape
    for group in landmarks:
        for landmark in group:
            frame = circle(
                frame,
                (int(landmark[0] * width), int(landmark[1] * height)),
                5,
                color,
                -1,
            )
        for connection in connections:
            start = group[connection[0]]
            end = group[connection[1]]
            frame = line(
                frame,
                (int(start[0] * width), int(start[1] * height)),
                (int(end[0] * width), int(end[1] * height)),
                color,
                2,
            )
    return frame
