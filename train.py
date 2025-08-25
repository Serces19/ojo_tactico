from ultralytics import YOLO

# Cargar el modelo
model = YOLO("yolov8n.pt")

# Entrenar
model.train(
    data="./roboflow/data.yaml", 
    epochs=50,
    imgsz=1280,
    batch=4,
    device="cpu",
    project="runs_yolo",
    name="yolov8n",
    workers=2,
    plots=True
)