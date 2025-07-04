from itertools import count
from camera import Camera
from audio import Audio
from window import Window
from detect import Detector
import measure


with Camera() as camera, Detector(camera.frame) as detector, Window(
    camera.frame, detector.players, detector.hands, detector.poses
) as window, Audio(
    detector.hands, detector.poses, window.inputs
), window.inputs.spawn() as receive:
    for index in count():
        inputs = receive.pop()
        if inputs.exit:
            break
        elif index % 1000 == 0:
            print(measure.flush())
