from __future__ import print_function
import cv2 as cv
import argparse
import time

# --------------------
# HSV threshold config
# --------------------
max_value = 255
max_value_H = 360 // 2

low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

# --------------------
# Trackbar callbacks
# --------------------
def on_low_H_thresh_trackbar(val):
    global low_H, high_H
    low_H = min(high_H - 1, val)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H, high_H
    high_H = max(low_H + 1, val)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S, high_S
    low_S = min(high_S - 1, val)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S, high_S
    high_S = max(low_S + 1, val)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V, high_V
    low_V = min(high_V - 1, val)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V, high_V
    high_V = max(low_V + 1, val)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

# --------------------
# Args / camera
# --------------------
parser = argparse.ArgumentParser(description='HSV Object Detection Playground')
parser.add_argument('--camera', default=0, type=int)
args = parser.parse_args()

cap = cv.VideoCapture(args.camera)

# --------------------
# Windows + trackbars
# --------------------
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.namedWindow("Heatmap Overlay")

cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

# --------------------
# Runtime toggles
# --------------------
use_morph = False
freeze_hsv = False

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
prev_time = time.time()

# --------------------
# Main loop
# --------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    frame_threshold = cv.inRange(
        frame_HSV,
        (low_H, low_S, low_V),
        (high_H, high_S, high_V)
    )

    if use_morph:
        frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel)
        frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(frame_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    object_count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 500:
            continue

        object_count += 1
        x, y, w, h = cv.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv.putText(frame, f"A:{int(area)}", (x, y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    # HUD
    cv.putText(frame, f"Objects: {object_count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(frame, f"FPS: {fps}", (10, 65),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(frame, "[m] morph  [f] freeze  [s] save  [q] quit",
               (10, frame.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    heatmap = cv.applyColorMap(frame_threshold, cv.COLORMAP_JET)
    overlay = cv.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    cv.imshow("Heatmap Overlay", overlay)

    key = cv.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break
    if key == ord('m'):
        use_morph = not use_morph
    if key == ord('f'):
        freeze_hsv = not freeze_hsv
    if key == ord('s'):
        filename = f"capture_H{low_H}-{high_H}_S{low_S}-{high_S}_V{low_V}-{high_V}.png"
        cv.imwrite(filename, frame)

cap.release()
cv.destroyAllWindows()
