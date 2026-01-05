#include <opencv2/opencv.hpp>

using namespace cv;

std::string imagePath = "macaw.jpg";

Mat loadImage(std::string imagepath, std::string colorSpace) {
    Mat result;
    Mat image = imread(imagePath);
    if (colorSpace == "gray") {
        cvtColor(image, result, COLOR_BGR2GRAY);
    } 
    else if (colorSpace == "rgb") {
        cvtColor(image, result, COLOR_BGR2RGB);
    }
    else if (colorSpace == "hsv") {
        cvtColor(image, result, COLOR_BGR2HSV);
    }
    else if (colorSpace == "bgr") {
        result = image;
    }
    return result;
}

void averageIntensities(Mat image) {
    
    cv::Scalar mean = cv::mean(image);
    // Print the average pixel values for each channel (BGR format)
    std::cout << "Average Blue Channel Value: " << mean[0] << std::endl;
    std::cout << "Average Green Channel Value: " << mean[1] << std::endl;
    std::cout << "Average Red Channel Value: " << mean[2] << std::endl;
}

int main() {
    Mat grayImage = loadImage(imagePath, "gray");
    Mat hsvImage = loadImage(imagePath, "hsv");
    Mat rgbImage = loadImage(imagePath, "rgb");
    Mat bgrImage = loadImage(imagePath, "bgr");
    //imshow("Gray", grayImage);
    //imshow("BGR", bgrImage);
    //waitKey();
    averageIntensities(bgrImage);
    return 0;
}