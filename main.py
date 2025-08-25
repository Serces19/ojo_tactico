import cv2
import os
import json
from tqdm import tqdm
from ultralytics import YOLO

def create_label_studio_annotation(image_filename, image_width, image_height, detections):
    """
    Crea una anotación en formato JSON compatible con Label Studio para una sola imagen.
    """
    annotations = {
        "annotations": [{
            "result": [],
            "id": "1",
            "lead_time": 0.0
        }],
        "data": {
            "image": f"/data/local-files/?d=dataset_curado/{image_filename}"
        }
    }

    for detection in detections:
        # Extraer las coordenadas del formato xywh
        x_center, y_center, width, height = detection["box_xywh"]

        # Convertir a formato xy top-left, width, height para Label Studio
        x_top_left = x_center - (width / 2)
        y_top_left = y_center - (height / 2)

        # Crear el objeto de resultado para la detección
        result_item = {
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "x": (x_top_left / image_width) * 100,
                "y": (y_top_left / image_height) * 100,
                "width": (width / image_width) * 100,
                "height": (height / image_height) * 100,
                "rotation": 0,
                "rectanglelabels": [detection["class_name"]]
            }
        }
        annotations["annotations"][0]["result"].append(result_item)
    
    return annotations

def main(images_dir):
    # --- Configuración de directorios ---
    # Cargar el modelo YOLOv8s desde la carpeta 'src'
    # Esta parte asume que el script está en el mismo nivel que el modelo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8s.pt')
    model = YOLO(model_path)

    # Definir rutas de salida relativas a la carpeta de imágenes
    base_dir = os.path.dirname(images_dir)
    annotations_dir = os.path.join(base_dir, 'anotaciones')
    visualizations_dir = os.path.join(base_dir, 'visualizaciones')

    # Crear directorios si no existen
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # --- Obtener la lista de imágenes ---
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {images_dir}")
        print("Asegúrate de colocar tus imágenes en ese directorio.")
        return

    # --- Procesar cada imagen ---
    for image_file in tqdm(image_files, desc="Procesando imágenes"):
        image_path = os.path.join(images_dir, image_file)
        
        # Leer la imagen
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen {image_file}. Saltando...")
            continue

        # Obtener dimensiones originales
        height, width, _ = frame.shape

        # --- Detección ---
        # No se aplica preprocesamiento para este ejemplo, ya que YOLO lo hace internamente
        results = model(frame, imgsz=640, classes=[0, 32], verbose=False)
        
        # --- Extracción de Datos y Creación de Anotaciones ---
        detections = []
        annotated_frame = frame.copy()

        # Iterar sobre los resultados de detección
        for res in results:
            annotated_frame = res.plot()  # Crea la imagen con los bounding boxes
            for box in res.boxes:
                detection_data = {
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box_xywh": box.xywh[0].tolist() # Convertir tensor a lista
                }
                detections.append(detection_data)
        
        # Crear la anotación en formato Label Studio
        annotation_data = create_label_studio_annotation(image_file, width, height, detections)
        
        # --- Guardar Anotación en JSON ---
        json_filename = os.path.splitext(image_file)[0] + ".json"
        json_path = os.path.join(annotations_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)
            
        # --- Guardar Visualización ---
        vis_filename = f"vis_{image_file}"
        vis_path = os.path.join(visualizations_dir, vis_filename)
        cv2.imwrite(vis_path, annotated_frame)

    print("\n¡Procesamiento de imágenes finalizado!")
    print(f"Anotaciones guardadas en: {os.path.abspath(annotations_dir)}")
    print(f"Visualizaciones guardadas en: {os.path.abspath(visualizations_dir)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Procesamiento de imágenes con YOLOv8 para generar anotaciones.")
    parser.add_argument('--images_dir', type=str, required=True, help="Ruta al directorio de imágenes.")
    args = parser.parse_args()
    
    # Asegurarse de que la ruta del directorio de entrada sea absoluta
    if not os.path.isabs(args.images_dir):
        args.images_dir = os.path.abspath(args.images_dir)

    main(args.images_dir)