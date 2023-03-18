//
// Created by bismarck on 23-3-13.
//

#ifndef DIGITALRECOGNITION_MODELMANAGER_H
#define DIGITALRECOGNITION_MODELMANAGER_H

#include <cstdint>
#include <atomic>
#include <deque>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "../InferResult.h"
#include "preprocess.h"

#define INPUT_MAT_TYPE CV_16UC1
#define INPUT_VAR_TYPE uint16_t

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


class ModelManager {
private:
    friend class InferResultAsync;
    friend class InferRequest;

#ifdef MOBILE_NET
    constexpr static const char* output_name = "874";
#endif
#ifdef BP
    constexpr static const char* output_name = "14";
#endif

    static std::deque<std::atomic<bool>> memoryUsing;
    static std::deque<INPUT_VAR_TYPE*> input_p;
    static std::deque<float*> preprocess_p;
    static std::deque<float*> output_p;

    constexpr static int output_size = 13;
    constexpr static int input_size = 32 * 32;

    static nvinfer1::IRuntime* runtime;
    static nvinfer1::ICudaEngine* engine;
    static std::deque<nvinfer1::IExecutionContext*> context_p;

    static InferResult postprocess(float* res);
    static void appendCache();

public:
    static void init();
    InferResult infer_sync(cv::Mat& img);
    InferResultAsync infer_async(cv::Mat& img);
};


#endif //DIGITALRECOGNITION_MODELMANAGER_H
