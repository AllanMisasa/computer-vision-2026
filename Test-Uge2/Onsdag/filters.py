import cv2
import numpy as np
import os

# -------------------------
# Load image
# -------------------------
img_path = "balls.jpg"
frame = cv2.imread(img_path)

if frame is None:
    raise FileNotFoundError("balls.jpg kunne ikke indlæses")

# Resize for stabil parametre
frame = cv2.resize(frame, (600, 400))

# -------------------------
# Preprocess
# -------------------------
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# -------------------------
# HSV range (til bolde i balls.jpg)
# Justér hvis nødvendigt
# -------------------------
lower = (0, 40, 40)
upper = (180, 255, 255)

mask = cv2.inRange(hsv, lower, upper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# -------------------------
# Find contours
# -------------------------
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = frame.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue

    ((x, y), radius) = cv2.minEnclosingCircle(c)

    if radius > 10:
        cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(output, (int(x), int(y)), 3, (0, 0, 255), -1)

# -------------------------
# Show results
# -------------------------
cv2.imshow("Mask", mask)
cv2.imshow("Detected balls", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
