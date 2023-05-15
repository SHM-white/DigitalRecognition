//
// Created by bismarck on 23-3-13.
//

#ifndef DIGITALRECOGNITION_MODELMANAGER_H
#define DIGITALRECOGNITION_MODELMANAGER_H

#define INIT_CACHE_SIZE 10

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include "../common.h"

class InferRequest;
class InferResult;
class InferResultAsync;

class ModelManager {
private:
    friend class InferResultAsync;
    friend class InferRequest;

    ov::Core core;

    std::deque<ov::InferRequest> requests;
    std::deque<ov::CompiledModel> models;
    std::deque<std::atomic<bool>> requestUsing;

    void scaleRequestPoll(int size=INIT_CACHE_SIZE);
    int getRequestSlot();

    ov::Tensor preprocess(cv::Mat& img);
    static InferResult postprocess(const ov::Tensor& res);

public:
    void init();
    InferResult infer_sync(cv::Mat& img);
    InferResultAsync infer_async(cv::Mat& img);
};


#endif //DIGITALRECOGNITION_MODELMANAGER_H
