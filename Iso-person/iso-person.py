import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os

INPUT = "IMG_0333.jpeg"
OUTPUT_DIR = "output"
OUTPUT_FILE = "isolated_person_cropped.png"

# Justér margin omkring personen (pixels)
PADDING = 20  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD + RESIZE
# =========================
bgr = cv2.imread(INPUT)
if bgr is None:
    raise RuntimeError("Image not loaded")

bgr = cv2.resize(bgr, (512, 768))

# =========================
# REMBG
# =========================
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)

result = remove(pil_img)          # RGBA PIL
rgba = np.array(result)           # RGBA numpy

# =========================
# FIND BOUNDING BOX (ALPHA)
# =========================
alpha = rgba[:, :, 3]

ys, xs = np.where(alpha > 0)
if len(xs) == 0 or len(ys) == 0:
    raise RuntimeError("No foreground detected")

x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

# Add padding (clamped)
h, w = alpha.shape
x_min = max(0, x_min - PADDING)
y_min = max(0, y_min - PADDING)
x_max = min(w, x_max + PADDING)
y_max = min(h, y_max + PADDING)

# =========================
# CROP
# =========================
cropped_rgba = rgba[y_min:y_max, x_min:x_max]

# =========================
# SAVE
# =========================
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
Image.fromarray(cropped_rgba).save(out_path)

print("Saved cropped image:", os.path.abspath(out_path))

# =========================
# VIS (DEBUG)
# =========================
bgr_vis = cv2.cvtColor(cropped_rgba, cv2.COLOR_RGBA2BGR)
cv2.imshow("Cropped person", bgr_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
