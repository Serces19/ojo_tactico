import cv2
import glob
import numpy as np

# Carpeta de imágenes y labels
img_paths = glob.glob("data_futbol/images/train/*.jpg")
label_paths = [p.replace("images", "labels").replace(".jpg", ".txt") for p in img_paths]

for img_path, lbl_path in zip(img_paths, label_paths):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Leer YOLO labels
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        # Convertir coordenadas normalizadas a pixeles
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key == 27:  # presiona ESC para salir
        break

cv2.destroyAllWindows()
