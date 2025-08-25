import cv2
import os
import json
from tqdm import tqdm
from ultralytics import YOLO
import numpy

def create_label_studio_annotation_pose(image_filename, image_width, image_height, detections):
    """
    Crea una anotación en formato JSON compatible con Label Studio para detección de pose.
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
        # Extraer las coordenadas del formato xywh para la caja delimitadora
        x_center_bbox, y_center_bbox, width_bbox, height_bbox = detection["box_xywh"]
        x_top_left_bbox = x_center_bbox - (width_bbox / 2)
        y_top_left_bbox = y_center_bbox - (height_bbox / 2)

        # Crear el objeto de resultado para la caja delimitadora
        bbox_item = {
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "x": (x_top_left_bbox / image_width) * 100,
                "y": (y_top_left_bbox / image_height) * 100,
                "width": (width_bbox / image_width) * 100,
                "height": (height_bbox / image_height) * 100,
                "rotation": 0,
                "rectanglelabels": ["person"]  # Asumimos que la clase 0 es 'person'
            }
        }
        annotations["annotations"][0]["result"].append(bbox_item)

        # Extraer los puntos clave (keypoints)
        keypoints_list = []
        for kp in detection["keypoints_xy"]:
            x_kp, y_kp = kp
            keypoints_list.append({
                "x": (x_kp / image_width) * 100,
                "y": (y_kp / image_height) * 100
            })

        # Crear el objeto de resultado para los puntos clave
        keypoints_item = {
            "from_name": "pose",
            "to_name": "image",
            "type": "keypointlabels",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "keypointlabels": ["keypoint"],
                "points": keypoints_list
            }
        }
        annotations["annotations"][0]["result"].append(keypoints_item)
    
    return annotations

def main(images_dir):
    # --- Configuración de directorios ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Cargar el modelo YOLOv8-Pose
    # Puedes descargar el modelo desde https://github.com/ultralytics/ultralytics
    model_path = os.path.join(script_dir, 'yolov8n-pose.pt')
    model = YOLO(model_path)

    # Definir rutas de salida relativas a la carpeta de imágenes
    base_dir = os.path.dirname(images_dir)
    annotations_dir = os.path.join(base_dir, 'anotaciones_pose')
    visualizations_dir = os.path.join(base_dir, 'visualizaciones_pose')

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
    for image_file in tqdm(image_files, desc="Procesando imágenes para pose"):
        image_path = os.path.join(images_dir, image_file)
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen {image_file}. Saltando...")
            continue

        height, width, _ = frame.shape

        # --- Detección de Pose ---
        results = model(frame, imgsz=640, verbose=False)
        
        # --- Extracción de Datos y Creación de Anotaciones ---
        detections = []
        annotated_frame = frame.copy()

        for res in results:
            annotated_frame = res.plot()  # Crea la imagen con la pose y los bounding boxes
            for i, box in enumerate(res.boxes):
                detection_data = {
                    "class_id": int(box.cls),
                    "class_name": "person",  # YOLOv8-Pose está preentrenado para 'person'
                    "confidence": float(box.conf),
                    "box_xywh": box.xywh[0].tolist(),
                    "keypoints_xy": res.keypoints.xy[i].tolist() if res.keypoints and len(res.keypoints.xy) > i else []
                }
                detections.append(detection_data)
        
        # Crear la anotación en formato Label Studio
        annotation_data = create_label_studio_annotation_pose(image_file, width, height, detections)
        
        # --- Guardar Anotación en JSON ---
        json_filename = os.path.splitext(image_file)[0] + ".json"
        json_path = os.path.join(annotations_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)
            
        # --- Guardar Visualización ---
        vis_filename = f"vis_{image_file}"
        vis_path = os.path.join(visualizations_dir, vis_filename)
        cv2.imwrite(vis_path, annotated_frame)

    print("\n¡Procesamiento de pose finalizado!")
    print(f"Anotaciones de pose guardadas en: {os.path.abspath(annotations_dir)}")
    print(f"Visualizaciones de pose guardadas en: {os.path.abspath(visualizations_dir)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Procesamiento de imágenes con YOLOv8 para generar anotaciones de pose.")
    parser.add_argument('--images_dir', type=str, required=True, help="Ruta al directorio de imágenes.")
    args = parser.parse_args()
    
    if not os.path.isabs(args.images_dir):
        args.images_dir = os.path.abspath(args.images_dir)

    main(args.images_dir)