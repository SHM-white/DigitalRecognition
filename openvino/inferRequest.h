//
// Created by bismarck on 23-5-15.
//

#ifndef SRC_INFERREQUEST_H
#define SRC_INFERREQUEST_H

#include <openvino/openvino.hpp>
#include "modelManager.h"

struct InferRequest {
    ModelManager* mm;
    int idx;
};

#endif //SRC_INFERREQUEST_H
