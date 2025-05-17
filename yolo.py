from ultralytics import YOLO


model = YOLO("models/yolo/yolo12l-pose.pt").cuda()
for result in model(source=0, stream=True, show=True):
    print(result)
