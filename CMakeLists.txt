cmake_minimum_required(VERSION 3.10)
project(DigitalRecognition)

set(CMAKE_CXX_STANDARD 14)
# 目前支持的识别方法：MobileNet, BP, Template
if (NOT RECOGNITION_METHOD)
    set(RECOGNITION_METHOD "MobileNet")
endif()
# 目前支持的后端引擎：openvino, tensorrt
if (NOT NN_METHOD)
    set(NN_METHOD "tensorrt")
endif()
set(BENCHMARK on)

function(define_openvino modelName)
    add_definitions(-DOPENVINO)
    add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/${modelName}.xml")

    set(OpenVINO_DIR /opt/intel/openvino/runtime/cmake)
    find_package(OpenVINO REQUIRED)

    set(SOURCE_FILES openvino/modelManager.cpp PARENT_SCOPE)
    set(INCLUDE_DIRS ${OpenVINO_INCLUDE_DIR} /opt/intel/openvino/runtime/include PARENT_SCOPE)
    set(LINK_LIBRARIES openvino::runtime PARENT_SCOPE)
endfunction(define_openvino)

function(define_tensorrt modelName)
    add_definitions(-DTENSORRT)
    add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/${modelName}.trtmodel")
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)

    set(CMAKE_CUDA_ARCHITECTURES native)

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message("embed_platform on")
        add_definitions(-DJETSON)
        include_directories(/usr/local/cuda/targets/aarch64-linux/include)
        link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
    else ()
        message("embed_platform off")
        # cuda
        include_directories(${CUDA_INCLUDE_DIRS})
        include_directories(/usr/local/cuda/include)
        link_directories(/usr/local/cuda/lib64)

        # tensorrt
        include_directories(/home/nvidia/TensorRT-8.2.5.1/include)
        link_directories(/home/nvidia/TensorRT-8.2.5.1/lib)
    endif ()

    cuda_add_library(cupp STATIC tensorrt/preprocess.cu)
    set(SOURCE_FILES tensorrt/modelManager.cpp tensorrt/inferRequest.cpp PARENT_SCOPE)
    set(LINK_LIBRARIES nvinfer ${CUDA_LIBRARIES} cupp PARENT_SCOPE)
endfunction(define_tensorrt)

if(${RECOGNITION_METHOD} MATCHES "MobileNet")
    message(STATUS "Using RECOGNITION METHOD MobileNet")
    add_definitions(-DMOBILE_NET)
    if(${NN_METHOD} MATCHES "openvino")
        define_openvino("mobileNet")
    elseif(${NN_METHOD} MATCHES "tensorrt")
        define_tensorrt("mobileNet")
    endif()
elseif(RECOGNITION_METHOD MATCHES "BP")
    message(STATUS "Using RECOGNITION METHOD BP")
    add_definitions(-DBP)
    if(${NN_METHOD} MATCHES "openvino")
        define_openvino("BP")
    elseif(${NN_METHOD} MATCHES "tensorrt")
        define_tensorrt("BP")
    endif()
elseif(RECOGNITION_METHOD MATCHES "Template")
    message(STATUS "Using RECOGNITION METHOD Template")
    add_definitions(-DTEMPLATES_PATH="${PROJECT_SOURCE_DIR}/templateMatch/templates/")
    add_definitions(-DTEMPLATE)

    set(SOURCE_FILES templateMatch/modelManager.cpp)
else()
    message(FATAL_ERROR "Not Define RECOGNITION METHOD")
endif()

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIR}
        ${INCLUDE_DIRS}
)

add_library(DigitalRecognitionLib STATIC ${SOURCE_FILES} InferResult.cpp)
target_link_libraries(DigitalRecognitionLib ${LINK_LIBRARIES} ${OpenCV_LIBRARIES})

if(BENCHMARK)
    add_executable(benchmark benchmark.cpp)
    target_link_libraries(benchmark DigitalRecognitionLib ${OpenCV_LIBRARIES})
endif()
