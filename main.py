from math import sqrt
from queue import Queue
from threading import Event, Thread
from typing import Any, Iterable, List, Optional, Tuple
from pyo import Server, Sine, Pan, pa_get_output_devices, midiToHz, hzToMidi
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    flip,
    imshow,
    cvtColor,
    COLOR_BGR2RGB,
    VideoCapture,
    destroyWindow,
    pollKey,
    namedWindow,
    resize,
    resizeWindow,
    WINDOW_NORMAL,
    CAP_PROP_POS_MSEC,
)
from cv2.typing import MatLike
from detect import Detector


WINDOW = "MEDIA PIPE"


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

    def frames(self) -> Iterable[Tuple[MatLike, int]]:
        while self._camera.isOpened():
            success, frame = self._camera.read(self._frame)
            self._frame = frame
            if success:
                frame = flip(frame, 1, frame)
                time = int(self._camera.get(CAP_PROP_POS_MSEC))
                yield frame, time
            else:
                break


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
        imshow(self._name, resize(frame, (self._width, self._height)))
        key = pollKey()
        change = key != self._last
        self._last = key
        return key, change


class Audio:
    MAJOR = [0, 0, 2, 2, 4, 5, 5, 7, 7, 9, 9, 11]
    MINOR = [0, 0, 2, 3, 3, 5, 5, 7, 8, 8, 10, 10]

    class Instrument:
        def __init__(self, scale: List[int]):
            self._play = Sine(freq=0, mul=0).out()
            # self._play = Pan(Sine(freq=0), mul=0).out()
            self._scale = scale

        def fade(self, volume: float, delta: float) -> bool:
            value = lerp(self._play.mul, volume, 10 * delta)
            self._play.mul = value
            return abs(volume - value) < 0.001

        def pan(self, pan: float, delta: float) -> bool:
            return True
            # value = lerp(self._play.pan, pan, 10 * delta)
            # self._play.pan = value
            # return abs(pan - value) < 0.001

        def shift(self, frequency: float):
            self._play.freq = note(frequency, self._scale)

    @staticmethod
    def device() -> Optional[int]:
        for name, index in zip(*pa_get_output_devices()):
            if "analog" in name.lower():
                return index

    @staticmethod
    def run(queue: Queue, abort: Event):
        _time = None
        _instruments: List[Audio.Instrument] = []

        while not abort.is_set():
            points, volume, time = queue.get()
            delta = 0 if _time is None else (time - _time) / 1000
            _time = time

            while len(_instruments) < len(points):
                _instruments.append(Audio.Instrument(Audio.MINOR))

            volume = sqrt(clamp(volume / (len(points) + 1)))
            for instrument, (x, y) in zip(_instruments, points):
                instrument.shift(y * 500)
                instrument.pan(x, delta)
                instrument.fade(volume, delta)

            for instrument in _instruments[len(points) :]:
                instrument.fade(0, delta)

    def __init__(self):
        self._server = Server(duplex=0, nchnls=2)
        self._server.setInOutDevice(Audio.device())
        self._server.boot().start()
        self._playing: List[Sine] = []
        self._fading: List[Sine] = []
        self._stopped: List[Sine] = []
        self._time = None
        self._queue = Queue(1)
        self._abort = Event()
        self._thread = Thread(target=Audio.run, args=(self._queue, self._abort))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._abort.set()
        self._server.stop()

    def update(self, points: List[Tuple[float, float]], volume: float, time: float):
        self._queue.put((points, volume, time))


def note(frequency: float, scale: List[int]) -> float:
    frequency = max(frequency, 1)
    midi = int(hzToMidi(frequency))
    degree = midi % len(scale)
    note = midi - degree + scale[degree]
    return midiToHz(note)


def clamp(value: float, minimum: float = 0, maximum: float = 1) -> float:
    return max(minimum, min(maximum, value))


def lerp(source: float, target: float, time: float) -> float:
    return source + (target - source) * clamp(time)


with Detector() as detector, Camera() as camera, Window() as window, Audio() as audio:
    success = True
    frame = None
    debug = False
    mute = False
    for frame, time in camera.frames():
        hands, poses = detector.detect(frame, time)

        if debug:
            frame = detector.draw(frame, hands, poses)

        key, change = window.show(frame)
        if change:
            if key == ord("d"):
                debug = not debug
            elif key == ord("m"):
                mute = not mute
            elif key in (ord("q"), 27):
                break

        audio.update(
            [
                finger
                for hand in hands
                for finger in (
                    hand[HandLandmark.THUMB_TIP],
                    hand[HandLandmark.INDEX_FINGER_TIP],
                    hand[HandLandmark.MIDDLE_FINGER_TIP],
                    hand[HandLandmark.RING_FINGER_TIP],
                    hand[HandLandmark.PINKY_TIP],
                )
            ],
            0 if mute else 1,
            time,
        )


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
