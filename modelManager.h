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

namespace dr {
    class _modelManager:public boost::noncopyable {
    private:
        ov::Core core;
        ov::CompiledModel model;

    public:
        void init();
        InferResult infer_sync(cv::Mat& img);
    };
}

#define modelManager boost::serialization::singleton<dr::_modelManager>::get_mutable_instance()

#endif //DIGITALRECOGNITION_MODELMANAGER_H
