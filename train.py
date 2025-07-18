from ultralytics import YOLO

model=YOLO("yolo11_custom.pt")  

model.train(data="data.yaml", epochs=30, imgsz=640, batch=8, workers=0, device=0)