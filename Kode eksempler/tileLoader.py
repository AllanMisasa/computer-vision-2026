import cv2 as cv
import os

def loadImage(imagePath, colorSpace):
    img = cv.imread(imagePath) # Load image
    if colorSpace == "gray":
        result = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale
    elif colorSpace == "rgb":
        result = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif colorSpace == "hsv":
        result = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elif colorSpace == "bgr":
        result = img
    else:
        print(f"Colorspace {colorSpace} is not configured.")
    return result

dir_path = "saved_patches/"
files = [dir_path + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
images = [loadImage(i, "bgr") for i in files]

cv.imshow("Patch", images[0])
cv.waitKey(0)
cv.destroyAllWindows()