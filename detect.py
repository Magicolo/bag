from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from posixpath import dirname, join
from typing import Iterable, List, Literal, Tuple, Union

from cv2 import COLOR_BGR2RGB, circle, cvtColor, line
from cv2.typing import MatLike, Scalar
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureRecognizerOptions,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.holistic_landmarker import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
)
from mediapipe.tasks.python.vision.image_segmenter import (
    ImageSegmenter,
    ImageSegmenterOptions,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from ultralytics import YOLO

Landmarks = List[Tuple[float, float]]
Connections = List[Tuple[int, int]]


class Detector:
    @staticmethod
    def _model_path(
        folder: Union[Literal["mediapipe"], Literal["yolo"]], name: str
    ) -> str:
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

    def __init__(self):
        self._pool = ThreadPoolExecutor()
        self._exit = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for exit in self._exit:
            exit()

    def detect(
        self, frame: MatLike, time: int
    ) -> Tuple[List[Landmarks], List[Landmarks]]:
        image = Image(ImageFormat.SRGB, cvtColor(frame, COLOR_BGR2RGB))
        hands = self._pool.submit(self._predict_mediapipe_hand, image, time)
        poses = self._pool.submit(self._predict_mediapipe_pose, image, time)
        # poses = self._predict_yolo_pose(frame)
        return hands.result(), poses.result()

    def draw(
        self,
        frame: MatLike,
        hands: List[Landmarks] = [],
        poses: List[Landmarks] = [],
    ) -> MatLike:
        frame = self._draw_landmarks(
            frame,
            (0, 255, 0),
            hands,
            [
                (connection.start, connection.end)
                for connection in HandLandmarksConnections.HAND_CONNECTIONS
            ],
        )
        frame = self._draw_landmarks(frame, (0, 0, 255), poses, [])
        return frame

    @cached_property
    def _mediapipe_holistic(self) -> HolisticLandmarker:
        holistic = HolisticLandmarker.create_from_options(
            HolisticLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=Detector._model_path(
                        "mediapipe", "holistic_landmarker.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
            )
        )
        self._exit.append(holistic.close)
        return holistic

    @cached_property
    def _mediapipe_hand(self) -> HandLandmarker:
        hand = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=Detector._model_path(
                        "mediapipe", "hand_landmarker.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=4,
            )
        )
        self._exit.append(hand.close)
        return hand

    @cached_property
    def _mediapipe_pose(self) -> PoseLandmarker:
        pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=Detector._model_path(
                        "mediapipe", "pose_landmarker_full.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
            )
        )
        self._exit.append(pose.close)
        return pose

    @cached_property
    def _mediapipe_gesture(self) -> GestureRecognizer:
        gesture = GestureRecognizer.create_from_options(
            GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=Detector._model_path(
                        "mediapipe", "gesture_recognizer.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=4,
            )
        )
        self._exit.append(gesture.close)
        return gesture

    @cached_property
    def _mediapipe_segment(self):
        segment = ImageSegmenter.create_from_options(
            ImageSegmenterOptions(
                base_options=BaseOptions(
                    model_asset_path=Detector._model_path(
                        "mediapipe", "selfie_segmentation.task"
                    ),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionTaskRunningMode.VIDEO,
            )
        )
        self._exit.append(segment.close)
        return segment

    def _predict_mediapipe_hand(self, image: Image, time: int) -> List[Landmarks]:
        hands = self._mediapipe_hand.detect_for_video(image, time)
        return [
            [(landmark.x, landmark.y) for landmark in hand]
            for hand in hands.hand_landmarks
        ]

    def _predict_mediapipe_pose(self, image: Image, time: int) -> List[Landmarks]:
        poses = self._mediapipe_pose.detect_for_video(image, time)
        return [
            [(landmark.x, landmark.y) for landmark in pose]
            for pose in poses.pose_landmarks
        ]

    def _predict_yolo_pose(self, frame: MatLike) -> Iterable[Landmarks]:
        for results in self._yolo_pose.predict(frame, stream=True):
            if results.keypoints:
                for pose in results.keypoints.xyn:
                    yield pose.tolist()

    @cached_property
    def _yolo_pose(self) -> YOLO:
        return YOLO(Detector._model_path("yolo", "yolo11n-pose.pt")).cuda()

    @cached_property
    def _yolo_object(self) -> YOLO:
        model = YOLO(Detector._model_path("yolo", "yolo12l.pt")).cuda()
        model.fuse()
        return model
