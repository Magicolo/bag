from typing import Iterable
from ultralytics import YOLO
import mediapipe
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    RunningMode,
    HolisticLandmarker,
    HandLandmarkerOptions,
)
from cv2 import (
    imshow,
    cvtColor,
    COLOR_BGR2RGB,
    VideoCapture,
    destroyWindow,
    pollKey,
    getWindowProperty,
    WND_PROP_VISIBLE,
    CAP_PROP_POS_MSEC,
)
from cv2.typing import MatLike


def yolo():
    model = YOLO("models/yolo11x-pose.pt").cuda()
    for result in model(source=0, stream=True):
        yield result.plot()


def frames(camera: VideoCapture) -> Iterable[MatLike]:
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            yield cvtColor(frame, COLOR_BGR2RGB)
        else:
            break


WINDOW = "YOLO"
camera = VideoCapture(0)
holistic = mediapipe.solutions.holistic
draw_landmarks = mediapipe.solutions.drawing_utils.draw_landmarks
try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/holistic_landmarker.task",
            model_asset_buffer=None,
            delegate=BaseOptions.Delegate.GPU,
        ),
        output_segmentation_mask=True,
        running_mode=RunningMode.VIDEO,
        running_mode=RunningMode.LIVE_STREAM,
        # result_callback=lambda result, image, _: None,
    )
    with HolisticLandmarker.create_from_options(options) as model:
        for frame in frames(camera):
            time = int(camera.get(CAP_PROP_POS_MSEC))
            image = Image.create_from_array(frame)
            result = model.detect_for_video(image, time)
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
            if exit():
                break
finally:
    camera.release()
    destroyWindow(WINDOW)


def exit() -> bool:
    pollKey() == 27 or getWindowProperty(WINDOW, WND_PROP_VISIBLE) < 1
