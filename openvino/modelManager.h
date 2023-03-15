//
// Created by bismarck on 23-3-13.
//

#ifndef DIGITALRECOGNITION_MODELMANAGER_H
#define DIGITALRECOGNITION_MODELMANAGER_H

#include <openvino/openvino.hpp>
#include <boost/serialization/singleton.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include "../InferResult.h"


class ModelManager {
private:
    friend class InferResultAsync;

#ifdef MOBILE_NET
    constexpr static const char* output_name = "874";
#endif
#ifdef BP
    constexpr static const char* output_name = "10";
#endif

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
