import cv2
from ultralytics import YOLO


model = YOLO('runs/detect/train/weights/best.pt')
vid = cv2.VideoCapture('c.mp4')

while True:
    ret, frame = vid.read()
    # frame = cv2.flip(frame, 1)
    results = model(frame)
    min_confidence = 0.8
    annotated_image = results[0].plot(conf=min_confidence)

    cv2.imshow("YOLOv8 Inference", annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
