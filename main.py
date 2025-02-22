from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLO('runs/detect/train6/weights/last.pt').to(device)
if __name__ =='__main__':
    model.train(data='mydata.yaml', epochs=70, imgsz=640, batch=40, optimizer='auto',workers=3, device='cuda',fliplr=0.0)
    metrics = model.val()