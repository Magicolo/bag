from ultralytics import YOLO
import cv2

model = YOLO("models/yolo11x-pose.pt")
for results in model(source=0, stream=True):
    frame = results.plot()
    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) == 27:
        break
