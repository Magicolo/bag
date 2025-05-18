from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from audio import Audio
from camera import Camera
from detect import Detector
from utility import clamp
from window import Window

FINGERS = [
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
]

with Audio() as audio, Camera() as camera, Window() as window, Detector() as detector:
    success = True
    frame = None
    debug = False
    mute = False
    for frame, time in camera.frames():
        hands, poses = detector.detect(frame, time)

        if debug:
            frame = detector.draw(frame, hands, poses)

        key, change = window.show(frame)
        if change:
            if key == ord("d"):
                debug = not debug
            elif key == ord("m"):
                mute = not mute
            elif key in (ord("q"), 27):
                break

        audio.update(
            [
                (clamp(x), clamp(1 - y) * 500 * (i + 1) + 100)
                for hand in hands
                for i, (x, y) in enumerate(hand[finger] for finger in FINGERS)
            ],
            0 if mute else 0.1,
            time,
        )


# ONNX:

# hand_detector = InferenceSession(
#     "models/mediapipe/hand_detector.onnx", providers=["CUDAExecutionProvider"]
# )
# hand_landmarks_detector = InferenceSession(
#     "models/mediapipe/hand_landmarks_detector.onnx", providers=["CUDAExecutionProvider"]
# )
# input: Any = hand_landmarks_detector.get_inputs()[0]
# if input is None:
#     raise ValueError("No input found.")
# outputs = [output.name for output in hand_landmarks_detector.get_outputs()][:2]

# image = numpy.expand_dims(
#     numpy.divide(
#         resize(cvtColor(small, COLOR_BGR2RGB), (WIDTH, HEIGHT)),
#         255.0,
#         dtype=numpy.float32,
#     ),
#     0,
# )
# height, width, _ = large.shape
# batches = iter(hand_detector.run(outputs, {input.name: image}))
# while True:
#     try:
#         for index, (box, confidence) in enumerate(
#             zip(next(batches).squeeze(0), next(batches).squeeze(0))
#         ):
#             x = box[0]
#             y = box[1]
#             w = box[2]
#             h = box[3]
#             if x >= 0 and y >= 0 and w >= 0 and h >= 0 and confidence > 1:
#                 large = rectangle(
#                     large,
#                     (int((x - w / 2) * width), int((y - h / 2) * height)),
#                     (int((x + w / 2) * width), int((y + h / 2) * height)),
#                     (0, 0, 255),
#                 )
#                 print(index, box[:4], confidence)
#     except StopIteration:
#         break

# for boxes in batch:
#     for box in boxes:
#         print(box)
#         x = box[0]
#         y = box[1]
#         w = box[2]
#         h = box[3]
#         large = rectangle(
#             large,
#             (int((x - w / 2) * width), int((y - h / 2) * height)),
#             (int((x + w / 2) * width), int((y + h / 2) * height)),
#             (0, 0, 255),
#         )
