from glob import glob
from typing import Tuple


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

from cell import Cells
import measure
from utility import run


class Camera:
    def __init__(self):
        self._frame = Cells[Tuple[MatLike, int]]()
        self._thread = run(self._run)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._frame.close()

    @property
    def frame(self) -> Cells[Tuple[MatLike, int]]:
        return self._frame

    def _run(self):
        _camera = _video(0, 1)
        try:
            _camera.set(CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(CAP_PROP_FRAME_HEIGHT, 480)
            _camera.set(CAP_PROP_FPS, 60)
            _camera.set(CAP_PROP_SHARPNESS, 5)
            _camera.set(CAP_PROP_GAMMA, 125)
            print(
                f"=> Camera Resolution: {int(_camera.get(CAP_PROP_FRAME_WIDTH))}x{int(_camera.get(CAP_PROP_FRAME_HEIGHT))}"
            )

            while _camera.isOpened():
                with measure.block("Camera"):
                    success, frame = _camera.read()
                    if success:
                        time = int(_camera.get(CAP_PROP_POS_MSEC))
                        frame = cvtColor(frame, COLOR_BGR2RGB, frame)
                        frame = flip(frame, 1, frame)
                        self._frame.set((frame, time))
                    else:
                        break
        finally:
            _camera.release()


def _video(*devices: str | int) -> VideoCapture:
    for device in iter((*devices, *sorted(glob("/dev/video*"), reverse=True))):
        print(f"=> Testing Camera: {device}")
        camera = VideoCapture(device)
        if camera.isOpened():
            print(f"=> Using Camera: {device}")
            return camera
        else:
            camera.release()
    raise ValueError("No camera found.")
