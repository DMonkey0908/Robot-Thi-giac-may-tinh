# import torch
from ultralytics import YOLO
import cv2
import Jetson.GPIO as GPIO
from time import sleep

# Load model
model = YOLO("best.pt")
GPIO.setmode(GPIO.BOARD)
# Load video
cap = cv2.VideoCapture(0)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)
# Loop over video frames
while True:
    # Read frame
    ret, frame = cap.read()
    # get the model names list
    names = model.names
    # Detect objects
    detections = model(frame)
    results = model(frame)
    # 
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
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(7, GPIO.HIGH)
        sleep(5)
        if object_count_1 > 0:
            GPIO.output(13, GPIO.LOW)
            print(f"Trai: {object_count_1}")
            GPIO.output(13, GPIO.HIGH)
            sleep(2)
        elif object_count_2 > 0:
            print(f"Phai: {object_count_2}")
            GPIO.output(7, GPIO.HIGH)
            sleep(2)
    elif object_count_3 > 0:
        print(f"Stop: {object_count_3}")
        GPIO.output(13, GPIO.LOW)
        GPIO.output(7, GPIO.LOW)
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

