//
// Created by bismarck on 23-3-13.
//

#include "modelManager.h"

#include <cstring>

namespace dr {
    void _modelManager::init() {
        model = core.compile_model(MODEL_PATH, "AUTO");
    }

    uint8_t _modelManager::infer_sync(cv::Mat &img) {
        ov::InferRequest infer_request = model.create_infer_request();
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
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape());
        memcpy(input_tensor.data<float_t>(), img.data, input_tensor.get_size());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto output = infer_request.get_tensor("873");
        const auto* output_buffer = output.data<const float>();
        return 0;
    }
}
