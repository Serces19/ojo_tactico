import cv2
import os
from ultralytics import YOLO


# Colores personalizados para clases (opcional)
colors = {
    0: (0, 255, 0),    # persona: verde
    1: (255, 0, 0),    # bicicleta: azul
    2: (0, 0, 255),    # coche: rojo
    # ... añade más según necesites
}

# Nombres personalizados para las etiquetas
custom_names = {
    0: "Bola",
    1: "PT",
    2: "Jugador",
    3: "Arbitro",
    # ... personaliza según tus necesidades
}


def main(model_path, images_dir, output_dir):
    """
    Realiza la inferencia de detección de pose en una carpeta de imágenes.
    """
    # Cargar el modelo entrenado
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que la ruta al archivo 'best.pt' sea correcta.")
        return
    

    # Obtener nombres de clases
    class_names = model.names

    # Mostrar todas las clases
    for idx, name in class_names.items():
        print(f"{idx}: {name}")

    # Definir el directorio de salida para las imágenes con anotaciones
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener la lista de imágenes
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {images_dir}")
        return

    print("Iniciando inferencia...")

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        
        # Leer la imagen
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen {image_file}. Saltando...")
            continue
            
        # Realizar la inferencia con el modelo
        results = model(
        frame,
        imgsz=1920,           # Tamaño de entrada. Mayor = más detalle (múltiplos de 32)
        conf=0.7,           # Umbral de confianza. Bajarlo detecta más objetos (incluso débiles)
        iou=0.3,            # Umbral de NMS. Menor = menos solapamiento, evita duplicados
        max_det=300,         # Máximo número de detecciones. Aumentarlo ayuda si hay muchos objetos
        half=False,          # Usa FP16 (más rápido, pero menor precisión). Pon False para máxima precisión
        device='cpu',       # Usa GPU si está disponible ('cuda') o 'cpu'
        augment=True,       # Aumentación en inferencia (TTA): mejora calidad, pero más lento
        agnostic_nms=False,  # Fusionar cajas de diferentes clases si se solapan
        classes=None,        # Filtrar por clases específicas (dejar None = todas)
        verbose=True         # Mostrar logs
    )

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)

                # Elegir nombre personalizado o usar el original
                label_name = custom_names.get(cls_id, result.names[cls_id])
                # Ejemplo de texto personalizado: "Auto (92%)"
                label = f"{label_name} ({conf:.0%})"

                # Color: personalizado o por defecto
                color = colors.get(cls_id, (255, 255, 0))  # amarillo por defecto

                # Dibujar caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Dibujar etiqueta con fondo
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


        # Extraer el frame con las anotaciones
        #annotated_frame = results[0].plot()

        # Guardar la imagen con las anotaciones
        output_path = os.path.join(output_dir, f"inf_{image_file}")
        cv2.imwrite(output_path, frame)
        print(f"Inferencia guardada para: {image_file}")

    print("\n¡Inferencia finalizada!")
    print(f"Las imágenes con las anotaciones se encuentran en: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script de inferencia de YOLOv8.")
    parser.add_argument('--images_dir', type=str, required=True, help="Ruta al directorio con las imágenes para inferencia.")
    parser.add_argument('--model_path', type=str, default="./runs_yolo/yolov8n18/weights/best.pt", help="Ruta al modelo .pt entrenado.")
    parser.add_argument('--output_dir', type=str, default="./inference_output", help="Directorio de salida para las imágenes con anotaciones.")
    args = parser.parse_args()

    main(args.model_path, args.images_dir, args.output_dir)