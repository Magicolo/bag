from typing import Any
from cv2 import COLOR_BGR2RGB, cvtColor, rectangle, resize
from onnxruntime import InferenceSession
import numpy

from camera import Camera
from utility import debug
from window import Window


# model = "models/mediapipe/hand_landmarks_detector.onnx"
model = "models/mediapipe/hand_detector.onnx"
session = InferenceSession(model, providers=["CUDAExecutionProvider"])
input: Any = next(iter(session.get_inputs()))
if input is None:
    raise ValueError("No input found.")
_, height, width, _ = debug(input.shape)
outputs = [output.name for output in session.get_outputs()][:2]

with Camera() as camera, Window() as window:
    for frame, time in camera.frames():
        sized_frame = resize(frame, (width, height))
        colored_frame = cvtColor(sized_frame, COLOR_BGR2RGB)
        divided_frame = numpy.divide(colored_frame, 255.0, dtype=numpy.float32)
        expanded_frame = numpy.expand_dims(divided_frame, 0)

        (n1, v1), (n2, v2) = zip(
            outputs, session.run(outputs, {input.name: expanded_frame})
        )
        for i, (s1, s2) in enumerate(zip(v1.squeeze(0), v2.squeeze(0))):
            if s2[0] > 0.95:
                x, y, w, h, *_ = s1[:]
                print(f"OUTPUT({i})", x, y, w, h)
                sized_frame = rectangle(
                    sized_frame,
                    (int(x - w / 2 + width / 2), int(y - h / 2 + height / 2)),
                    (int(x + w / 2 + width / 2), int(y + h / 2 + height / 2)),
                    1,
                )

        window.show(sized_frame)
