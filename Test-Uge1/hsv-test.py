import cv2 as cv
import time

# --------------------
# Camera
# --------------------
cap = cv.VideoCapture(0)

# --------------------
# Face detector (Haar)
# --------------------
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------
# Person detector (HOG + SVM)
# --------------------
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# --------------------
# UI Window
# --------------------
cv.namedWindow("Controls")

def nothing(x):
    pass

# --------------------
# Trackbars (LIVE PARAMETERS)
# --------------------
cv.createTrackbar("Face scale x100", "Controls", 120, 200, nothing)
cv.createTrackbar("Face neighbors", "Controls", 5, 20, nothing)

cv.createTrackbar("Min person area", "Controls", 3000, 20000, nothing)
cv.createTrackbar("Min aspect x100", "Controls", 150, 300, nothing)

cv.createTrackbar("ROI start %", "Controls", 30, 80, nothing)
cv.createTrackbar("Frame skip", "Controls", 3, 10, nothing)

prev_time = time.time()
frame_id = 0

# --------------------
# Main loop
# --------------------
while True:
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # --------------------
    # Read parameters
    # --------------------
    face_scale = cv.getTrackbarPos("Face scale x100", "Controls") / 100
    face_neighbors = max(1, cv.getTrackbarPos("Face neighbors", "Controls"))

    min_area = cv.getTrackbarPos("Min person area", "Controls")
    min_ratio = cv.getTrackbarPos("Min aspect x100", "Controls") / 100

    roi_start = cv.getTrackbarPos("ROI start %", "Controls") / 100
    frame_skip = max(1, cv.getTrackbarPos("Frame skip", "Controls"))

    # ====================
    # FACE DETECTION
    # ====================
    face_boxes = []
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=face_scale,
        minNeighbors=face_neighbors,
        minSize=(40, 40)
    )

    for (fx, fy, fw, fh) in faces:
        face_boxes.append((fx, fy, fw, fh))
        cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv.putText(frame, "Face", (fx, fy - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ====================
    # PERSON DETECTION
    # ====================
    boxes = []
    if frame_id % frame_skip == 0:
        roi_y = int(h_img * roi_start)
        roi = frame[roi_y:, :]

        raw_boxes, _ = hog.detectMultiScale(
            roi,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        for (x, y, w, h) in raw_boxes:
            if w * h < min_area:
                continue

            ratio = h / w
            if ratio < min_ratio:
                continue

            y += roi_y
            boxes.append((x, y, w, h))

    # ====================
    # DRAW PERSONS + FACE LOGIC
    # ====================
    for (x, y, w, h) in boxes:
        has_face = False
        for (fx, fy, fw, fh) in face_boxes:
            if fx > x and fy > y and fx + fw < x + w and fy + fh < y + h:
                has_face = True
                break

        color = (0, 255, 0) if has_face else (255, 0, 0)
        label = "Person+Face" if has_face else "Person"

        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, label, (x, y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ====================
    # FPS
    # ====================
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    cv.putText(frame, f"FPS: {fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Face + Person Detection", frame)

    if cv.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv.destroyAllWindows()
