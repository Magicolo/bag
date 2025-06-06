from typing import Tuple

from cv2 import (
    COLOR_RGB2BGR,
    WINDOW_NORMAL,
    cvtColor,
    destroyWindow,
    imshow,
    namedWindow,
    pollKey,
    resize,
    resizeWindow,
)
from cv2.typing import MatLike

import measure


class Window:
    def __init__(self, name="La Brousse À Gigante", width=640, height=480):
        self._name = name
        self._width = width
        self._height = height
        self._last = None
        namedWindow(name, WINDOW_NORMAL)
        resizeWindow(name, width, height)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        destroyWindow(self._name)

    def show(self, frame: MatLike) -> Tuple[int, bool]:
        with measure.block("Window"):
            frame = cvtColor(frame, COLOR_RGB2BGR, frame)
            frame = resize(frame, (self._width, self._height), frame)
            imshow(self._name, frame)
            key = pollKey()
            change = key != self._last
            self._last = key
            return key, change
