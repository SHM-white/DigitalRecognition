//
// Created by bismarck on 23-3-18.
//

#ifndef DIGITALRECOGNITION_INFERREQUEST_H
#define DIGITALRECOGNITION_INFERREQUEST_H

#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
class InferResult;

class InferRequest {
private:
    cudaStream_t stream{nullptr};
    float* output{nullptr};
    int idx{-1};
public:
    InferRequest() = default;
    InferRequest(cudaStream_t _stream, int _idx);
    InferRequest(const InferRequest&) = delete;
    InferRequest(InferRequest&&) = default;
    InferRequest& operator=(InferRequest &&other) noexcept;
    ~InferRequest();
    InferResult get();
};

typedef std::shared_ptr<InferRequest> InferRequestPtr;

#endif //DIGITALRECOGNITION_INFERREQUEST_H
