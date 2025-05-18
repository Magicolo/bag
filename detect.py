from functools import cached_property
from posixpath import dirname, join
from queue import Queue
from threading import Thread
from typing import Iterable, List, Literal, Tuple, Union
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

from utility import magnitude, multiply

Landmarks = List[Tuple[float, float]]
Connections = List[Tuple[int, int]]


class Landmark:
    def __init__(self):
        self._position = 0, 0, 0
        self._motion = 0, 0, 0
        self._visibility = 0
        self._presence = 0

    @property
    def x(self):
        return self._position[0]

    @property
    def y(self):
        return self._position[1]

    @property
    def z(self):
        return self._position[2]

    @property
    def position(self):
        return self._position

    @property
    def motion(self):
        return self._motion

    @property
    def visibility(self):
        return self._visibility

    @property
    def presence(self):
        return self._presence

    def velocity(self, delta: float) -> Tuple[float, float, float]:
        return multiply(self._motion, delta)

    def speed(self, delta: float) -> float:
        return magnitude(self.velocity(delta))

    def set(self, landmark: NormalizedLandmark):
        x, y, z = landmark.x or 0, landmark.y or 0, landmark.z or 0
        self._motion = x - self.x, y - self.y, z - self.z
        self._position = x, y, z
        self._visibility = landmark.visibility or 0
        self._presence = landmark.presence or 0

    def clear(self):
        self._position = 0, 0, 0
        self._motion = 0, 0, 0
        self._visibility = 0
        self._presence = 0


class Hand:
    def __init__(self):
        self._landmarks = [Landmark() for _ in range(21)]
        self._handedness = 0

    @property
    def handedness(self) -> float:
        """
        Returns:
            float: -1 is left handed, 1 is right handed
        """
        return self._handedness

    @property
    def landmarks(self) -> List[Landmark]:
        return self._landmarks

    @property
    def finger_tips(self) -> Tuple[Landmark, Landmark, Landmark, Landmark, Landmark]:
        return (
            self._landmarks[HandLandmark.THUMB_TIP],
            self._landmarks[HandLandmark.INDEX_FINGER_TIP],
            self._landmarks[HandLandmark.MIDDLE_FINGER_TIP],
            self._landmarks[HandLandmark.RING_FINGER_TIP],
            self._landmarks[HandLandmark.PINKY_TIP],
        )

    @property
    def finger_bases(self) -> Tuple[Landmark, Landmark, Landmark, Landmark, Landmark]:
        return (
            self._landmarks[HandLandmark.THUMB_MCP],
            self._landmarks[HandLandmark.INDEX_FINGER_MCP],
            self._landmarks[HandLandmark.MIDDLE_FINGER_MCP],
            self._landmarks[HandLandmark.RING_FINGER_MCP],
            self._landmarks[HandLandmark.PINKY_MCP],
        )

    def set(self, landmarks: List[NormalizedLandmark], handedness: Category):
        if handedness.index == 0:
            self._handedness = handedness.score or 0
        elif handedness.index == 1:
            self._handedness = -(handedness.score or 0)
        else:
            self._handedness = 0

        for landmark, normalized in zip(self._landmarks, landmarks):
            landmark.set(normalized)

    def clear(self):
        self._handedness = 0
        for landmark in self._landmarks:
            landmark.clear()


class Detector:
    def __init__(self):
        self._hands = Queue(1), Queue(1)
        self._poses = Queue(1), Queue(1)
        self._threads = (
            Thread(target=_run_hands, args=self._hands),
            Thread(target=_run_poses, args=self._poses),
        )
        for thread in self._threads:
            thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hands[0].put(None)
        self._poses[0].put(None)
        for thread in self._threads:
            thread.join(1)

    def detect(self, frame: MatLike, time: int) -> Tuple[List[Hand], List[Landmarks]]:
        image = Image(ImageFormat.SRGB, cvtColor(frame, COLOR_BGR2RGB))
        self._hands[0].put((image, time))
        self._poses[0].put((image, time))
        return self._hands[1].get(), self._poses[1].get()

    def draw(
        self,
        frame: MatLike,
        hands: List[Landmarks] = [],
        poses: List[Landmarks] = [],
    ) -> MatLike:
        frame = _draw_landmarks(
            frame,
            (0, 255, 0),
            hands,
            [
                (connection.start, connection.end)
                for connection in HandLandmarksConnections.HAND_CONNECTIONS
            ],
        )
        frame = _draw_landmarks(
            frame,
            (0, 0, 255),
            poses,
            [
                (connection.start, connection.end)
                for connection in PoseLandmarksConnections.POSE_LANDMARKS
            ],
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


def _run_hands(receive: Queue, send: Queue):
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

    _hands: List[Hand] = []
    with load() as model:
        while True:
            message = receive.get()
            if message is None:
                break

            image, time = message
            result = model.detect_for_video(image, time)
            while len(_hands) < len(result.hand_landmarks):
                _hands.append(Hand())

            for hand, landmarks, handedness in zip(
                _hands, result.hand_landmarks, result.handedness
            ):
                hand.set(landmarks, handedness[0])

            for hand in _hands[len(result.hand_landmarks) :]:
                hand.clear()

            send.put(_hands[: len(result.hand_landmarks)])


def _run_poses(receive: Queue, send: Queue):
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
    landmarks: List[Landmarks],
    connections: Connections,
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
