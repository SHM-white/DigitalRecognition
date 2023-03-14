//
// Created by bismarck on 23-3-13.
//

#ifndef DIGITALRECOGNITION_MODELMANAGER_H
#define DIGITALRECOGNITION_MODELMANAGER_H

#include <openvino/openvino.hpp>
#include <boost/serialization/singleton.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>

struct InferResult {
    int id;
    float confidence;
};

class InferResultAsync {
private:
    ov::InferRequest req;
public:
    InferResultAsync() = default;
    InferResultAsync(const InferResultAsync& other) = default;
    explicit InferResultAsync(ov::InferRequest&& _req);
    InferResultAsync(InferResultAsync&& other) = default;
    InferResultAsync& operator=(const InferResultAsync& other);
    InferResult get();
};

class ModelManager: public boost::noncopyable {
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
