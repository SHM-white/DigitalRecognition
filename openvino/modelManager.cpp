//
// Created by bismarck on 23-3-13.
//

#include "modelManager.h"
#include "../tensorrt/modelManager.h"


#include <cstring>

InferResult datasetId2InferResult[] = {
        {1, true, 1},
        {2, false, 1},
        {3, false, 1},
        {3, true, 1},
        {4, false, 1},
        {4, true, 1},
        {5, false, 1},
        {5, true, 1},
        {6, true, 1},
        {7, false, 1},
        {8, false, 1},
        {8, true, 1},
        {0, false, 0},
};

void ModelManager::init() {
    model = core.compile_model(MODEL_PATH, "AUTO");
}

ov::Tensor ModelManager::preprocess(cv::Mat &img) {
    auto input_port = model.input();
    if(img.channels() > 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    if (img.rows != height || img.cols != width) {
        cv::resize(img, img, cv::Size(width, height));
    }
    if (img.type() != CV_32FC1) {
        img.convertTo(img, CV_32FC1);
    }
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img.ptr(0));
    return input_tensor;
}

InferResult ModelManager::postprocess(const ov::Tensor& output) {
    auto shape = output.get_shape();
    const auto* output_buffer = output.data<const float>();
    InferResult maxRes{255, false, 0};
    for(int i = 0; i < (int)shape[1] - 1; i++) {
        if (maxRes.id == 255 || output_buffer[i] > maxRes.confidence) {
            maxRes = datasetId2InferResult[i];
            maxRes.confidence = output_buffer[i];
        }
    }
    return maxRes;
}

InferResult ModelManager::infer_sync(cv::Mat &img) {
    auto input_tensor = preprocess(img);
    ov::InferRequest infer_request = model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_tensor(output_name);
    return postprocess(output);
}

InferResultAsync ModelManager::infer_async(cv::Mat &&img) {
    auto input_tensor = preprocess(img);
    ov::InferRequest infer_request = model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    return InferResultAsync{std::move(infer_request)};
}
