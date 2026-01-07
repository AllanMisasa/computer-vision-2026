from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# =========================
# GLOBALS
# =========================
src = None
src_gray = None

max_elem = 3
max_kernel_size = 21
max_thresh = 255

window_controls = "Controls"
window_result = "Contours"

# =========================
# MORPH SHAPE
# =========================
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE
    elif val == 3:
        return cv.MORPH_DIAMOND

# =========================
# MAIN UPDATE FUNCTION
# =========================
def update(val=0):
    # --- Trackbar values ---
    erosion_size = cv.getTrackbarPos("Erosion size", window_controls)
    dilation_size = cv.getTrackbarPos("Dilation size", window_controls)
    morph_type = cv.getTrackbarPos("Element", window_controls)
    canny_thresh = cv.getTrackbarPos("Canny thresh", window_controls)

    shape = morph_shape(morph_type)

    # --- Structuring elements ---
    erosion_element = cv.getStructuringElement(
        shape,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size)
    )

    dilation_element = cv.getStructuringElement(
        shape,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size)
    )

    # --- Morphology ---
    morphed = cv.erode(src_gray, erosion_element)
    morphed = cv.dilate(morphed, dilation_element)

    # --- Canny ---
    canny = cv.Canny(morphed, canny_thresh, canny_thresh * 2)

    # --- Find contours ---
    contours, hierarchy = cv.findContours(
        canny,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )

    # --- Draw contours ---
    drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    # --- Show ---
    cv.imshow("Morphed (Erode + Dilate)", morphed)
    cv.imshow("Canny", canny)
    cv.imshow(window_result, drawing)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="IMG_0333.jpeg", help="Path to image")
    args = parser.parse_args()

    src = cv.imread(args.input)
    if src is None:
        print("Could not load image")
        exit(0)

    src = cv.resize(src, (400, 600))
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.GaussianBlur(src_gray, (3, 3), 0)

    # --- Windows ---
    cv.namedWindow(window_controls)
    cv.namedWindow(window_result)

    # --- Trackbars ---
    cv.createTrackbar("Element", window_controls, 0, max_elem, update)
    cv.createTrackbar("Erosion size", window_controls, 1, max_kernel_size, update)
    cv.createTrackbar("Dilation size", window_controls, 1, max_kernel_size, update)
    cv.createTrackbar("Canny thresh", window_controls, 100, max_thresh, update)

    # --- Initial render ---
    update()

    cv.imshow("Original", src)
    cv.waitKey()
    cv.destroyAllWindows()
