//
// Created by bismarck on 23-3-13.
//

#include "modelManager.h"
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("../test.png");
    img = 255 - img;
    modelManager.init();
    modelManager.infer_sync(img);
}
