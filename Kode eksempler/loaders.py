import cv2 as cv

imagePath = "CPP/Computer Vision/2025 January/macaw.jpg"

def loadImage(imagePath, colorSpace):
    img = cv.imread(imagePath) # Load image
    if colorSpace == "gray":
        result = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale
    elif colorSpace == "rgb":
        result = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif colorSpace == "hsv":
        result = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    else:
        print(f"Colorspace {colorSpace} is not configured.")
    return result


def main():
    grayImg = loadImage(imagePath, "gray") 
    rgbImg = loadImage(imagePath, "rgb")
    hsvImg = loadImage(imagePath, "hsv")
    bgrImg = cv.imread(imagePath)
    cv.imshow('Grayscale', grayImg) # Display the grayscale image
    cv.imshow('RGB', rgbImg)
    cv.imshow('BGR', bgrImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

main()