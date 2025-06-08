from itertools import count
from camera import Camera
from audio import Audio
from cell import Cell
from window import Window
from detect import Detector
import measure

with Cell(None) as stop, Camera(stop) as camera, Detector(
    camera.frame, stop
) as detector, Window(
    camera.frame, detector.players, detector.hands, detector.poses, stop
) as window, Audio(
    detector.players, window.inputs, stop
), window.inputs.spawn() as receive:
    for index in count():
        inputs = receive.pop()
        if inputs.exit:
            stop.close()
            break
        elif index % 10 == 0:
            measure.flush()
