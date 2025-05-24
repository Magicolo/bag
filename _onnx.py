import time
from typing import Any
from cv2 import COLOR_BGR2RGB, VideoCapture, cvtColor, resize
from onnxruntime import InferenceSession
import numpy

from utility import debug

model = "models/mediapipe/hand_landmarks_detector.onnx"
# model = "models/mediapipe/hand_detector.onnx"
session = InferenceSession(model, providers=["CUDAExecutionProvider"])
input: Any = next(iter(session.get_inputs()))
if input is None:
    raise ValueError("No input found.")
_, height, width, _ = debug(input.shape)
outputs = [output.name for output in session.get_outputs()][:2]

index = 0
then = time.time()
frame = None
camera = VideoCapture(0)
try:
    while camera.isOpened():
        success, frame = camera.read(frame)
        if success:
            image = numpy.expand_dims(
                numpy.divide(
                    resize(cvtColor(frame, COLOR_BGR2RGB), (width, height)),
                    255.0,
                    dtype=numpy.float32,
                ),
                0,
            )
            for hand in session.run(outputs, {input.name: image}):
                print(hand)

            index += 1
            print(f"FPS: {index / (time.time() - then):.2f}")
finally:
    camera.release()
