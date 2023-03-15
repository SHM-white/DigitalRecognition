//
// Created by bismarck on 23-3-15.
//

#include "InferResult.h"

#ifdef OPENVINO
#include "openvino/modelManager.h"
#endif

InferResultAsync::InferResultAsync(InferResultAsync::T&& _req): req(_req) {}

InferResultAsync &InferResultAsync::operator=(const InferResultAsync &other) {
    if (&other == this) {
        return *this;
    }
    req = other.req;
    return *this;
}

#ifdef OPENVINO
InferResult InferResultAsync::get() {
    req.wait();
    return ModelManager::postprocess(req.get_tensor(ModelManager::output_name));
}
#endif

#ifdef TEMPLATE
InferResult InferResultAsync::get() {
    return req.get();
}
#endif
