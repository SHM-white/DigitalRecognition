//
// Created by bismarck on 23-3-18.
//
#include "modelManager.h"
#include <fstream>
#include "logging.h"

#define INIT_CACHE_SIZE 10

Logger glogger;

InferResult ModelManager::postprocess(float *res) {
    return {};
}

void ModelManager::preprocess(cv::Mat &img, int idx, cudaStream_t stream) {
    if(img.channels() > 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    if (img.rows != 32 || img.cols != 32) {
        cv::resize(img, img, cv::Size(32, 32));
    }
    if (img.type() != INPUT_MAT_TYPE) {
        img.convertTo(img, INPUT_MAT_TYPE);
    }
    memcpy(input_p[idx], img.data, input_size * sizeof(INPUT_VAR_TYPE));
    CHECK(cuMemPrefetchAsync((CUdeviceptr)input_p[idx], input_size * sizeof(INPUT_VAR_TYPE), device, stream));
    cuda_preprocess(input_p[idx], preprocess_p[idx], input_size, stream);
}

void ModelManager::init() {
    cudaGetDevice(&device);
    char* trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(MODEL_PATH, std::ios::binary);
    if (file.good()) {
        file.seekg(0, std::ifstream::end);
        size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, (long)size);
        file.close();
    }
    runtime = nvinfer1::createInferRuntime(glogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    appendCache();
}

void ModelManager::appendCache() {
    for (int i = 0; i < INIT_CACHE_SIZE; i++) {
        memoryUsing.emplace_back(false);
        INPUT_VAR_TYPE* input_;
        CHECK(cudaMallocManaged((void**)&input_, 32 * 32 * sizeof(INPUT_VAR_TYPE)));
        input_p.push_back(input_);
        float* preprocess_;
        CHECK(cudaMallocManaged((void**)&preprocess_, input_size * sizeof(float)));
        preprocess_p.push_back(preprocess_);
        float* output_;
        CHECK(cudaMallocManaged((void**)&output_, output_size * sizeof(float)));
        output_p.push_back(output_);
        auto context_ = engine->createExecutionContext();
        assert(context_ != nullptr);
        context_p.push_back(context_);
    }
}

InferResult ModelManager::infer_sync(cv::Mat& img) {
    return {};
}

InferResultAsync ModelManager::infer_async(cv::Mat& img) {
    return {};
}
