from dataclasses import dataclass
import threading
from typing import Callable, Generic, Iterable, List, TypeVar
from pyo import Server, Sine, pa_list_devices
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmark,
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureRecognizerOptions,
)
from mediapipe.tasks.python.vision.image_segmenter import (
    ImageSegmenter,
    ImageSegmenterOptions,
)
from mediapipe.tasks.python.vision import RunningMode
from cv2.typing import MatLike, Scalar
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    circle,
    flip,
    imshow,
    cvtColor,
    COLOR_BGR2RGB,
    VideoCapture,
    destroyWindow,
    line,
    pollKey,
    namedWindow,
    resize,
    resizeWindow,
    WINDOW_NORMAL,
    CAP_PROP_POS_MSEC,
)
from torch import device


T = TypeVar("T")
U = TypeVar("U")


class Lock(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value = value
        self._lock = threading.Lock()

    def lock(self, lock: Callable[[T], U]) -> U:
        with self._lock:
            return lock(self._value)


@dataclass
class Hand:
    landmarks: List

    def set(self, landmarks: List):
        self.landmarks = landmarks


@dataclass
class Pose:
    landmarks: List

    def set(self, landmarks: List):
        self.landmarks = landmarks


@dataclass
class State:
    hand: Hand
    pose: Pose


WINDOW = "MEDIA PIPE"
WIDTH = 80
HEIGHT = 60
SCALE = 8
MODE = RunningMode.LIVE_STREAM
DEVICE = BaseOptions.Delegate.GPU
STATE = Lock(State(Hand([]), Pose([])))

camera = VideoCapture(0)
camera.set(CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(CAP_PROP_FPS, 30)
namedWindow(WINDOW, WINDOW_NORMAL)
resizeWindow(WINDOW, WIDTH * SCALE, HEIGHT * SCALE)


def hand() -> HandLandmarker:
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/hand_landmarker.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
            num_hands=4,
            result_callback=lambda result, _, __: STATE.lock(
                lambda state: state.hand.set(result.hand_landmarks)
            ),
        )
    )


def pose() -> PoseLandmarker:
    return PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/pose_landmarker_full.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
            result_callback=lambda result, _, __: STATE.lock(
                lambda state: state.pose.set(result.pose_landmarks)
            ),
        )
    )


def gesture() -> GestureRecognizer:
    return GestureRecognizer.create_from_options(
        GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/gesture_recognizer.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
            num_hands=4,
        )
    )


def segment():
    return ImageSegmenter.create_from_options(
        ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/selfie_segmentation.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
        )
    )


def draw(
    frame: MatLike, color: Scalar, landmarks: Iterable[List], connections: Iterable
) -> MatLike:
    height, width, _ = frame.shape
    for landmarks in landmarks:
        for landmark in landmarks:
            frame = circle(
                frame,
                (int(landmark.x * width), int(landmark.y * height)),
                5,
                color,
                -1,
            )
        for connection in connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            frame = line(
                frame,
                (int(start.x * width), int(start.y * height)),
                (int(end.x * width), int(end.y * height)),
                color,
                2,
            )
    return frame


try:
    print(pa_list_devices())
    server = Server(duplex=0, nchnls=2)
    server.setInOutDevice(4)
    server.boot().start()
    carrier = Sine(freq=[400, 500, 600, 700, 800]).out()
    with hand() as hand_detector, pose() as pose_detector:
        success = True
        small = None
        large = None
        while success and camera.isOpened():
            success, small = camera.read(small)
            time = int(camera.get(CAP_PROP_POS_MSEC))
            small = flip(small, 1, small)
            image = Image(ImageFormat.SRGB, cvtColor(small, COLOR_BGR2RGB))
            hand_detector.detect_async(image, time)
            pose_detector.detect_async(image, time)
            (hand_landmarks, pose_landmarks) = STATE.lock(
                lambda state: (state.hand.landmarks, state.pose.landmarks)
            )
            large = resize(small, (WIDTH * SCALE, HEIGHT * SCALE), large)
            large = draw(
                large,
                (0, 0, 255),
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )
            large = draw(
                large,
                (0, 255, 0),
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
            )
            imshow(WINDOW, large)
            for landmarks in hand_landmarks:
                carrier.freq = [
                    frequency
                    for (index, landmark) in enumerate(
                        (
                            landmarks[HandLandmark.THUMB_TIP],
                            landmarks[HandLandmark.INDEX_FINGER_TIP],
                            landmarks[HandLandmark.MIDDLE_FINGER_TIP],
                            landmarks[HandLandmark.RING_FINGER_TIP],
                            landmarks[HandLandmark.PINKY_TIP],
                        ),
                        1,
                    )
                    for frequency in (
                        landmark.x * 440 * index,
                        landmark.y * 440 * index,
                    )
                ]
                print(carrier.freq)
            if pollKey() == 27:
                break
finally:
    camera.release()
    destroyWindow(WINDOW)
