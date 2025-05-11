from ultralytics import YOLO


model = YOLO("models/yolo11x-pose.pt").cuda()
for result in model(source=0, stream=True, show=True):
    pass
