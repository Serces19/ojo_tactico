import cv2
import os
from ultralytics import YOLO

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
        results = model(frame, verbose=False)

        # Extraer el frame con las anotaciones
        annotated_frame = results[0].plot()

        # Guardar la imagen con las anotaciones
        output_path = os.path.join(output_dir, f"inf_{image_file}")
        cv2.imwrite(output_path, annotated_frame)
        print(f"Inferencia guardada para: {image_file}")

    print("\n¡Inferencia finalizada!")
    print(f"Las imágenes con las anotaciones se encuentran en: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script de inferencia de YOLOv8.")
    parser.add_argument('--images_dir', type=str, required=True, help="Ruta al directorio con las imágenes para inferencia.")
    parser.add_argument('--model_path', type=str, default="./runs_yolo/yolov8n/weights/best.pt", help="Ruta al modelo .pt entrenado.")
    parser.add_argument('--output_dir', type=str, default="./inference_output", help="Directorio de salida para las imágenes con anotaciones.")
    args = parser.parse_args()

    main(args.model_path, args.images_dir, args.output_dir)