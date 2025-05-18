from typing import Iterable, Tuple

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_MSEC,
    VideoCapture,
    flip,
)
from cv2.typing import MatLike


class Camera:
    def __init__(self):
        self._camera = VideoCapture(0)
        self._camera.set(CAP_PROP_FRAME_WIDTH, 1)
        self._camera.set(CAP_PROP_FRAME_HEIGHT, 1)
        self._camera.set(CAP_PROP_FPS, 30)
        self._frame = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._camera.release()

    def frames(self) -> Iterable[Tuple[MatLike, int, int]]:
        _time = None
        while self._camera.isOpened():
            success, frame = self._camera.read(self._frame)
            self._frame = frame
            if success:
                frame = flip(frame, 1, frame)
                time = int(self._camera.get(CAP_PROP_POS_MSEC))
                delta = 0 if _time is None else time - _time
                _time = time
                yield frame, time, delta
            else:
                break
