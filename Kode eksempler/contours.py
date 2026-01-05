import cv2 as cv

imagePath = "CPP/Computer Vision/Basics/macaw.jpg"

def loadImage(imagePath: str, colorSpace: str) -> cv.Mat:
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

def blurFilter(img: cv.Mat, kernelSize: int, sigma: int, type: str) -> cv.Mat:
    if type == "gauss":
        blurred = cv.GaussianBlur(img, (kernelSize, kernelSize), 0)
    elif type == "bilateral":
        blurred = cv.bilateralFilter(img, kernelSize, sigma, sigma)
    else: 
        print(f"Filter of type {type} is not supported.")
    return blurred

def contourDetection(img: cv.Mat) -> list:
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")
    return contours

def colorFilter(img: cv.Mat) -> cv.Mat:
    ...

def main() -> None: 
    grayImg = loadImage(imagePath, colorSpace="gray") 
    rgbImg = loadImage(imagePath, colorSpace="rgb")
    hsvImg = loadImage(imagePath, colorSpace="hsv")
    bgrImg = cv.imread(imagePath)
    #blurred = blurFilter(grayImg, kernelSize=11, sigma=75, type="bilateral")
    blurred = cv.GaussianBlur(grayImg, (5, 5), 2)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    cl1 = clahe.apply(blurred)
    # blurred = blurFilter(img=cl1, kernelSize=5, sigma=75, type="bilateral")
    contours = contourDetection(img=cl1)
    cv.drawContours(grayImg, contours, -1, (0,255,0), 3)
    cv.imshow(winname="Orginal", mat=bgrImg)
    cv.imshow(winname="Contours on grayscale image", mat=grayImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

main()