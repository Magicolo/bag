from ultralytics import YOLO
import cv2
from cv2 import VideoCapture

for size in ["n", "s", "m", "l", "x"]:
    for kind in ["", "-pose", "-seg", "-obb", "-cls"]:
        YOLO(f"yolo11{size}{kind}.pt").export(format="onnx")
        
for size in ["n", "s", "m", "l", "x"]:
    YOLO(f"yolo12{size}.pt").export(format="onnx")
