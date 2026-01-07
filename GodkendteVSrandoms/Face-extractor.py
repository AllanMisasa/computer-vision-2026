import cv2
from rembg import remove
import os
import numpy as np

# --------------------
# Config
# --------------------
BASE_DIR = "GodkendteVSrandoms"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMG_SIZE = 160

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Face detector
# --------------------
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if cascade.empty():
    raise RuntimeError("Cannot load Haar cascade")

# --------------------
# Process images
# --------------------
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(in_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        continue

    # vælg største ansigt
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    pad = int(0.15 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    face = img[y1:y2, x1:x2]

    # fjern baggrund
    face_rgba = remove(face)
    face_rgb = face_rgba[:, :, :3]

    # standardiser størrelse
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, face_rgb)

print("Face extraction finished")
