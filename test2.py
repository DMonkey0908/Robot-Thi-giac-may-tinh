from PIL import Image
from ultralytics import YOLO
import cv2
# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

# Predict with the model
# results = model(['a.jpg','b.jpg','c.jpg'])  # predict on an image
results = model('b.jpg')  # predict on an image
for r in results:
    print(r.boxes)
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()
    r.labels()

# from PIL import Image
# from ultralytics import YOLO
# import cv2
# import pandas as pd
#
# # Load a model (replace with your path)
# # model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model
#
# # Predict on an image
# image_path = 'b.jpg'  # Replace with your image path
# results = model(image_path)
#
# # Prepare empty list for detections
# detections = []
#
# for det in results.pandas().xyxy[0].to_dict(orient='records'):
#     # Extract relevant information from each detection
#     label_name = det['name']
#     confidence = det['conf']
#     x_min, y_min, x_max, y_max = det['xmin'], det['ymin'], det['xmax'], det['ymax']
#
#     # Add detection details to the list
#     detections.append({
#         "Label": label_name,
#         "Confidence": confidence,
#         "X_Min": x_min,
#         "Y_Min": y_min,
#         "X_Max": x_max,
#         "Y_Max": y_max
#     })
#
# # Create a Pandas DataFrame from the detections list
# df = pd.DataFrame(detections)
#
# # Print the DataFrame to terminal
# print(df.to_string(index=False))
#
# # Optional: Visualization with OpenCV
# img = results[0].plot(labels=True, conf=True)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
