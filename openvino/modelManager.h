//
// Created by bismarck on 23-3-13.
//

#ifndef DIGITALRECOGNITION_MODELMANAGER_H
#define DIGITALRECOGNITION_MODELMANAGER_H

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include "../InferResult.h"
#include "../common.h"


class ModelManager {
private:
    friend class InferResultAsync;

    ov::Core core;
    ov::CompiledModel model;

    ov::Tensor preprocess(cv::Mat& img);
    static InferResult postprocess(const ov::Tensor& res);

public:
    void init();
    InferResult infer_sync(cv::Mat& img);
    InferResultAsync infer_async(cv::Mat& img);
};


#endif //DIGITALRECOGNITION_MODELMANAGER_H
