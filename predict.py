from ultralytics import YOLO

model=YOLO("yolo11_custom.pt")  

model.predict(source="0", show=True, save=True, conf=0.5)