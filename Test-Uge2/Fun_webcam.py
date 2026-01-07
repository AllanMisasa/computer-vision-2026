import cv2
import numpy as np

# Haar Cascade til ansigt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kan ikke åbne webcam")
    exit()

# Højere opløsning for webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

final_size = 512  # større canvas
canvas_bg = np.zeros((final_size, final_size, 3), dtype=np.uint8)  # sort baggrund

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Tag største ansigt
        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        face_crop = frame[y:y+h, x:x+w]

        # Resize til canvas og centrer
        fh, fw = face_crop.shape[:2]
        scale = min(final_size / fh, final_size / fw)
        new_size = (int(fw*scale), int(fh*scale))
        face_resized = cv2.resize(face_crop, new_size, interpolation=cv2.INTER_LINEAR)

        # Lav sort canvas og indsæt ansigtet centreret
        canvas = canvas_bg.copy()
        x_offset = (final_size - new_size[0]) // 2
        y_offset = (final_size - new_size[1]) // 2
        canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = face_resized

        cv2.imshow("Live Face Only", canvas)
    else:
        # Ingen ansigt → vis sort
        cv2.imshow("Live Face Only", canvas_bg)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
