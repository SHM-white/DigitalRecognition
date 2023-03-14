//
// Created by bismarck on 23-3-13.
//

#include "modelManager.h"

#include <cstring>

void ModelManager::init() {
    model = core.compile_model(MODEL_PATH, "AUTO");
}

ov::Tensor ModelManager::preprocess(cv::Mat &img) {
    auto input_port = model.input();
    if(img.channels() > 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    if (img.rows != 32 || img.cols != 32) {
        cv::resize(img, img, cv::Size(32, 32));
    }
    if (img.type() != CV_32FC1) {
        img.convertTo(img, CV_32FC1);
    }
    img -= 116.28;
    img /= 57.12;
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.ptr(0));
    return input_tensor;
}

InferResult ModelManager::postprocess(const ov::Tensor& output) {
    auto shape = output.get_shape();
    const auto* output_buffer = output.data<const float>();
    InferResult maxRes{255, 0};
    for(int i = 0; i < (int)shape[1]; i++) {
        if (maxRes.id == 255 || output_buffer[i] > maxRes.confidence) {
            maxRes = InferResult{i, output_buffer[i]};
        }
    }
    return maxRes;
}

InferResult ModelManager::infer_sync(cv::Mat &img) {
    auto input_tensor = preprocess(img);
    ov::InferRequest infer_request = model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_tensor("873");
    return postprocess(output);
}

InferResultAsync ModelManager::infer_async(cv::Mat &img) {
    auto input_tensor = preprocess(img);
    ov::InferRequest infer_request = model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    return InferResultAsync{std::move(infer_request)};
}

InferResultAsync::InferResultAsync(ov::InferRequest &&_req): req(_req) {}

InferResult InferResultAsync::get() {
    req.wait();
    return ModelManager::postprocess(req.get_tensor("873"));
}

InferResultAsync &InferResultAsync::operator=(const InferResultAsync &other) {
    if (&other == this) {
        return *this;
    }
    req = other.req;
    return *this;
}

