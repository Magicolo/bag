from typing import Iterable, List
from pyo import Server, Sine, pa_list_devices
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
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


WINDOW = "MEDIA PIPE"
WIDTH = 80
HEIGHT = 60
SCALE = 8
MODE = VisionTaskRunningMode.VIDEO
DEVICE = BaseOptions.Delegate.GPU

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
    server = Server(duplex=0, nchnls=2)
    server.setInOutDevice(4)
    server.boot().start()
    carrier = Sine(freq=[444, 555, 666, 777, 888]).out()
    with hand() as hand_detector, pose() as pose_detector:
        success = True
        small = None
        large = None
        while success and camera.isOpened():
            success, small = camera.read(small)
            time = int(camera.get(CAP_PROP_POS_MSEC))
            small = flip(small, 1, small)
            image = Image(ImageFormat.SRGB, cvtColor(small, COLOR_BGR2RGB))
            hand_result = hand_detector.detect_for_video(image, time)
            pose_result = pose_detector.detect_for_video(image, time)
            large = resize(small, (WIDTH * SCALE, HEIGHT * SCALE), large)
            large = draw(
                large,
                (0, 0, 255),
                hand_result.hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )
            large = draw(
                large,
                (0, 255, 0),
                pose_result.pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
            )
            for landmarks in hand_result.hand_landmarks:
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
                        landmark.x * 333 * index,
                        landmark.y * 555 * index,
                    )
                ]
            imshow(WINDOW, large)
            if pollKey() == 27:
                break
finally:
    camera.release()
    destroyWindow(WINDOW)
