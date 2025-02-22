import torch
from ultralytics import YOLO
import cv2
from time import sleep

# Load model
model = YOLO("runs/detect/train5/weights/best.pt")

# Load video
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('e.mp4')
# cap = model.predict(source="1")
# Loop over video frames
while True:
    # Read frame
    ret, frame = cap.read()
    # get the model names list
    names = model.names
    # Detect objects
    detections = model(frame)
    results = model(frame)
    # get the 'DiThang' class id
    DiThang = list(names)[list(names.values()).index('DiThang')]
    Trai = list(names)[list(names.values()).index('Trai')]
    Phai = list(names)[list(names.values()).index('Phai')]
    Stop = list(names)[list(names.values()).index('STOP')]
    # Count objects
    # count 'car' objects in the results
    object_count_0 = results[0].boxes.cls.tolist().count(DiThang)
    object_count_1 = results[0].boxes.cls.tolist().count(Trai)
    object_count_2 = results[0].boxes.cls.tolist().count(Phai)
    object_count_3 = results[0].boxes.cls.tolist().count(Stop)
    if object_count_0 > 0:
        print(f"DiThang: {object_count_0}")
        sleep(5)
    elif object_count_1 > 0:
        print(f"Trai: {object_count_1}")
        sleep(5)
    elif object_count_2 > 0:
        print(f"Phai: {object_count_2}")
        sleep(5)
    elif object_count_3 > 0:
        print(f"Stop: {object_count_3}")
        sleep(5)
    # object_count = len(detections)
    # print(f"DiThang: {object_count}")
    # Display results
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

