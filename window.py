from typing import Tuple

from cv2 import (
    WINDOW_NORMAL,
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
            imshow(self._name, resize(frame, (self._width, self._height)))
            key = pollKey()
            change = key != self._last
            self._last = key
            return key, change
