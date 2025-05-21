from typing import Iterable
from audio import Audio, Notes, Sound
from camera import Camera
from detect import Detector, Gesture, Hand
from utility import clamp, cut
from window import Window


def _sounds(hands: Iterable[Hand], mute: bool) -> Iterable[Sound]:
    volume = 0.0 if mute else 1.0
    for hand in hands:
        love = hand.gesture == Gesture.ILOVEYOU
        floor = 0 if love else 0.025
        triangle = any(hand.triangle(other) for other in hands)
        for index, finger in enumerate(hand.fingers):
            speed = cut(finger.tip.speed, floor)
            yield Sound(
                frequency=clamp(1 - finger.tip.y) * 1000.0 * (index % 5 + 1) + 50.0,
                amplitude=clamp(speed * 10.0 * volume),
                pan=clamp(finger.tip.x),
                notes=(Notes.SECRET if triangle else Notes.NATURAL),
                sequence=triangle,
                advance=speed,
                glide=0.25 if love else 0.025,
            )


with Audio() as audio, Camera() as camera, Window() as window, Detector() as detector:
    success = True
    frame = None
    show = False
    mute = False
    for frame, time in camera.frames():
        hands, poses = detector.detect(frame, time)

        if show:
            frame = detector.draw(frame, hands, poses)

        reset = False
        key, change = window.show(frame)
        if change:
            if key == ord("d"):
                show = not show
            elif key == ord("r"):
                reset = True
            elif key == ord("m"):
                mute = not mute
            elif key in (ord("q"), 27):
                break

        volume = 0.0 if mute else 1.0
        audio.send(_sounds(hands, mute), reset)


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
