from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
from time import sleep

model = YOLO('runs/detect/train7/weights/best.pt')
results = model.predict(source="1", show=True)

for detection  in model:
    label = detection['label']
    print(f"Detected object: {label}")
    sleep(1)