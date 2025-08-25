from ultralytics import YOLO

# Cargar el modelo
model = YOLO("yolov8s.pt")

# Entrenar
model.train(
    data="./roboflow/data.yaml", 
    epochs=50,
    imgsz=1280,
    batch=8,
    device="cpu",
    project="runs_yolo",
    name="yolov8s",
    workers=4,
    plots=True
)