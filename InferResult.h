//
// Created by bismarck on 23-3-15.
//

#ifndef DIGITALRECOGNITION_INFERRESULT_H
#define DIGITALRECOGNITION_INFERRESULT_H

#ifdef OPENVINO
#include <openvino/openvino.hpp>
#endif
#ifdef TEMPLATE
#include <future>
#endif

struct InferResult {
    int id;
    bool big;
    float confidence;
};

class InferResultAsync {
private:
#ifdef OPENVINO
    ov::InferRequest req;
#endif
#ifdef TEMPLATE
    std::shared_future<InferResult> req;
#endif
    using T = decltype(req);
    InferResult result{};
    bool callable = false, called = false, set = false;
public:
    InferResultAsync() = default;
    InferResultAsync(const InferResultAsync& other) = default;
    InferResultAsync(InferResultAsync&& other) = default;
    explicit InferResultAsync(T&& _req);
    InferResultAsync& operator=(const InferResultAsync& other);
    InferResult get();
    InferResult operator()();
    void setMarkerType(bool type);
};


#endif //DIGITALRECOGNITION_INFERRESULT_H
