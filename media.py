from typing import Iterable

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    RunningMode,
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    GestureRecognizer,
    GestureRecognizerOptions,
    ImageSegmenter,
    ImageSegmenterOptions,
)
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    imshow,
    cvtColor,
    COLOR_BGR2RGB,
    VideoCapture,
    destroyWindow,
    pollKey,
    getWindowProperty,
    WND_PROP_VISIBLE,
    namedWindow,
    resizeWindow,
    WINDOW_NORMAL,
    CAP_PROP_POS_MSEC,
)


WINDOW = "MEDIA PIPE"
WIDTH = 320
HEIGHT = 240
MODE = RunningMode.VIDEO
DEVICE = BaseOptions.Delegate.GPU
camera = VideoCapture(0)
camera.set(CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(CAP_PROP_FPS, 30)
namedWindow(WINDOW, WINDOW_NORMAL)
resizeWindow(WINDOW, WIDTH, HEIGHT)


def hand():
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/hand_landmarker.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
        )
    )


def pose():
    return PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/pose_landmarker_full.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
        )
    )


def gesture():
    return GestureRecognizer.create_from_options(
        GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="models/mediapipe/gesture_recognizer.task",
                delegate=DEVICE,
            ),
            running_mode=MODE,
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


try:
    with hand() as hand_detector, pose() as pose_detector, gesture() as gesture_recognizer:
        # with HolisticLandmarker.create_from_options(options) as model:
        success = True
        frame = None
        while success and camera.isOpened():
            success, frame = camera.read(frame)
            time = int(camera.get(CAP_PROP_POS_MSEC))
            image = Image(ImageFormat.SRGB, cvtColor(frame, COLOR_BGR2RGB))
            hand_result = hand_detector.detect_for_video(image, time)
            pose_result = pose_detector.detect_for_video(image, time)
            gesture_result = gesture_recognizer.recognize_for_video(image, time)
            print(f"HAND: {hand_result}")
            print(f"POSE: {pose_result}")
            print(f"GESTURE: {gesture_result}")
            # model.detect_async(image, time)
            # results = model.process(frame)
            # draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS)
            # draw_landmarks(frame, results.face_landmarks, holistic.FACEMESH_TESSELATION)
            # draw_landmarks(
            #     frame, results.left_hand_landmarks, holistic.HAND_CONNECTIONS
            # )
            # draw_landmarks(
            #     frame, results.right_hand_landmarks, holistic.HAND_CONNECTIONS
            # )
            imshow(WINDOW, frame)
            if pollKey() == 27:
                break
finally:
    camera.release()
    destroyWindow(WINDOW)
