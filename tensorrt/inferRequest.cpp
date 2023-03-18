//
// Created by bismarck on 23-3-18.
//

#include "inferRequest.h"
#include "../InferResult.h"
#include "modelManager.h"

InferRequest::InferRequest(cudaStream_t _stream, int _idx) {
    idx = _idx;
    ModelManager::memoryUsing[_idx] = true;
    stream = _stream;
    output = ModelManager::output_p[_idx];
}

InferRequest::~InferRequest() {
    if (output != nullptr) {
        output = nullptr;
        ModelManager::memoryUsing[idx] = false;
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

InferResult InferRequest::get() {
    if (stream == nullptr || output == nullptr) {
        return {0, false, 0};
    }
    cudaStreamSynchronize(stream);
    return ModelManager::postprocess(output);
}

InferRequest& InferRequest::operator=(InferRequest &&other) noexcept {
    if (&other == this) {
        return *this;
    }
    stream = other.stream;
    other.stream = nullptr;
    output = other.output;
    other.output = nullptr;
    idx = other.idx;
    other.idx = -1;
    return *this;
}
