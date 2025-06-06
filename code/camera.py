from glob import glob
from typing import Iterable, Tuple


from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_GAMMA,
    CAP_PROP_POS_MSEC,
    CAP_PROP_SHARPNESS,
    COLOR_BGR2RGB,
    VideoCapture,
    cvtColor,
    flip,
)
from cv2.typing import MatLike

from cell import Cell
import measure
from utility import run


class Camera:
    def __init__(self):
        self._cell = Cell[Tuple[MatLike, int]]()
        self._thread = run(_actor, self._cell)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cell.close()
        self._thread.join()

    def frames(self) -> Iterable[Tuple[MatLike, int]]:
        while True:
            yield self._cell.pop()


def _actor(cell: Cell[Tuple[MatLike, int]]):
    _device = None
    _camera = None
    for device in iter(sorted(glob("/dev/video*"), reverse=True)):
        print(f"=> Testing Camera: {device}")
        camera = VideoCapture(device)
        if camera.isOpened():
            _device = device
            _camera = camera
            break
        else:
            camera.release()
    if _device is None or _camera is None:
        raise ValueError("No camera found.")

    print(f"=> Using Camera: {_device}")
    try:
        _camera.set(CAP_PROP_FRAME_WIDTH, 640)
        _camera.set(CAP_PROP_FRAME_HEIGHT, 480)
        _camera.set(CAP_PROP_FPS, 60)
        _camera.set(CAP_PROP_SHARPNESS, 5)
        _camera.set(CAP_PROP_GAMMA, 125)
        print(
            f"=> Camera Resolution: {int(_camera.get(CAP_PROP_FRAME_WIDTH))}x{int(_camera.get(CAP_PROP_FRAME_HEIGHT))}"
        )

        for _ in measure.loop("Camera", _camera.isOpened):
            success, frame = _camera.read()
            if success:
                time = int(_camera.get(CAP_PROP_POS_MSEC))
                frame = cvtColor(frame, COLOR_BGR2RGB, frame)
                frame = flip(frame, 1, frame)
                cell.set((frame, time))
            else:
                break
    finally:
        _camera.release()
