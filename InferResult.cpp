//
// Created by bismarck on 23-3-15.
//

#include "InferResult.h"

#ifdef OPENVINO
#include "openvino/modelManager.h"
#endif

InferResultAsync::InferResultAsync(InferResultAsync::T&& _req): req(_req), callable(true) {}

InferResultAsync &InferResultAsync::operator=(const InferResultAsync &other) {
    if (&other == this) {
        return *this;
    }
    req = other.req;
    return *this;
}

InferResult InferResultAsync::operator()() {
    if (called) {
        return result;
    } else {
        if (callable) {
            result = get();
            called = true;
            callable = false;
            return result;
        } else {
            throw std::runtime_error("Call invalid InferResultAsync!");
        }
    }
}

#ifdef OPENVINO
InferResult InferResultAsync::get() {
    req.wait();
    InferResult res = ModelManager::postprocess(req.get_tensor(ModelManager::output_name));
    req = ov::InferRequest();
    return res;
}
#endif

#ifdef TEMPLATE
InferResult InferResultAsync::get() {
    return req.get();
}
#endif
