//
// Created by bismarck on 23-3-13.
//

#include <iostream>

#include <opencv2/opencv.hpp>

#include "modelManager.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"


int main() {
    cv::Mat img = cv::imread("../test.png");
    modelManager.init();
    InferResult res{};
    ankerl::nanobench::Bench().run("OpenVINO mobileNetV3 Small", [&res, &img] {
        res = modelManager.infer_sync(img);
    });
    std::cout << "Id: " << res.id << std::endl << "Confidence: " << res.confidence << std::endl;
}
