import cv2
import numpy as np

# =========================
# LOAD & RESIZE
# =========================
bgr = cv2.imread("../images/IMG_0334.jpeg")
bgr = cv2.resize(bgr, (400, 600))
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# =========================
# CALLBACK
# =========================
def nothing(x):
    pass

# =========================
# WINDOW + TRACKBARS
# =========================
cv2.namedWindow("HSV Controls")

cv2.createTrackbar("H min", "HSV Controls", 0, 179, nothing)
cv2.createTrackbar("H max", "HSV Controls", 18, 179, nothing)
cv2.createTrackbar("S min", "HSV Controls", 47, 255, nothing)
cv2.createTrackbar("S max", "HSV Controls", 255, 255, nothing)
cv2.createTrackbar("V min", "HSV Controls", 2, 255, nothing)
cv2.createTrackbar("V max", "HSV Controls", 255, 255, nothing)

# =========================
# MAIN LOOP
# =========================
while True:
    # --- Hent thresholds ---
    h_min = cv2.getTrackbarPos("H min", "HSV Controls")
    h_max = cv2.getTrackbarPos("H max", "HSV Controls")
    s_min = cv2.getTrackbarPos("S min", "HSV Controls")
    s_max = cv2.getTrackbarPos("S max", "HSV Controls")
    v_min = cv2.getTrackbarPos("V min", "HSV Controls")
    v_max = cv2.getTrackbarPos("V max", "HSV Controls")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # --- HSV mask ---
    mask = cv2.inRange(hsv, lower, upper)

    # --- Morfologi (stabilisering) ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Apply ---
    result = cv2.bitwise_and(bgr, bgr, mask=mask)

    # --- Vis ---
    cv2.imshow("HSV mask", mask)
    cv2.imshow("Fremhævet", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
