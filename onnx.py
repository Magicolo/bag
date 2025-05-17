import itertools
import time
from typing import Any
from onnxruntime import InferenceSession
import numpy

hand_detector = InferenceSession(
    "models/mediapipe/hand_detector.onnx", providers=["CUDAExecutionProvider"]
)
hand_landmarks_detector = InferenceSession(
    "models/mediapipe/hand_landmarks_detector.onnx", providers=["CUDAExecutionProvider"]
)
input: Any = next(iter(hand_landmarks_detector.get_inputs()))
if input is None:
    raise ValueError("No input found.")
output: Any = next(iter(hand_landmarks_detector.get_outputs()))
if output is None:
    raise ValueError("No output found.")

then = time.time()
for frame in itertools.count():
    for hand in hand_landmarks_detector.run(
        [output.name], {input.name: numpy.zeros(input.shape, dtype=numpy.float32)}
    ):
        pass
        # print(hand.shape)
    if frame % 100 == 0:
        print(f"FPS: {frame / (time.time() - then):.2f}")
