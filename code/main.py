from audio import Audio
from camera import Camera
from detect import Detector
import measure
from window import Window

with Audio() as audio, Camera() as camera, Window() as window, Detector.gpu() as detector:
    success = True
    frame = None
    show = False
    mute = False
    for index, (frame, time) in enumerate(camera.frames()):
        hands, poses = detector.detect(frame, time)

        if show:
            frame = detector.draw(frame, hands, poses)

        reset = False
        key, change = window.show(frame)
        if change:
            if key == ord("d"):
                show = not show
            elif key == ord("r"):
                reset = True
            elif key == ord("m"):
                mute = not mute
            elif key in (ord("q"), 27):
                break

        audio.send(hands, mute, reset)

        if index % 10 == 0:
            measure.flush()
