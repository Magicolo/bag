from threading import Thread
from typing import Iterable, Tuple


from cv2 import (
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_GAMMA,
    CAP_PROP_POS_MSEC,
    CAP_PROP_SHARPNESS,
    VideoCapture,
    flip,
)
from cv2.typing import MatLike

from channel import Channel, Closed
import measure
from utility import catch


class Camera:
    def __init__(self):
        self._channel = Channel[Tuple[MatLike, int]]()
        self._thread = Thread(target=catch(_actor, Closed, ()), args=(self._channel,))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._channel.close()
        self._thread.join()

    def frames(self) -> Iterable[Tuple[MatLike, int]]:
        while True:
            yield self._channel.get()


def _actor(channel: Channel[Tuple[MatLike, int]]):
    _camera = VideoCapture(0)
    try:
        _camera.set(CAP_PROP_FRAME_WIDTH, 320)
        _camera.set(CAP_PROP_FRAME_HEIGHT, 240)
        _camera.set(CAP_PROP_SHARPNESS, 7)
        _camera.set(CAP_PROP_GAMMA, 150)

        for _ in measure.loop("Camera", _camera.isOpened):
            success, frame = _camera.read()
            if success:
                frame = flip(frame, 1, frame)
                time = int(_camera.get(CAP_PROP_POS_MSEC))
                channel.put((frame, time))
            else:
                break
    finally:
        _camera.release()
