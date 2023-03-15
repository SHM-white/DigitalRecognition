cmake_minimum_required(VERSION 3.10)
project(DigitalRecognition)

set(CMAKE_CXX_STANDARD 14)
# 目前支持的识别方法：MobileNet, BP, Template
set(RECOGNITION_METHOD "BP")
if(${RECOGNITION_METHOD} MATCHES "MobileNet")
    message(STATUS "Using RECOGNITION METHOD MobileNet")
    add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/mobileNet.xml")
    add_definitions(-DMOBILE_NET)
    add_definitions(-DOPENVINO)

    set(OpenVINO_DIR /opt/intel/openvino/runtime/cmake)
    find_package(OpenVINO REQUIRED)

    set(SOURCE_FILES openvino/modelManager.cpp)
    set(INCLUDE_DIRS ${OpenVINO_INCLUDE_DIR} /opt/intel/openvino/runtime/include)
    set(LINK_LIBRARIES openvino::runtime)
elseif(RECOGNITION_METHOD MATCHES "BP")
    message(STATUS "Using RECOGNITION METHOD BP")
    add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/BP.xml")
    add_definitions(-DBP)
    add_definitions(-DOPENVINO)

    set(OpenVINO_DIR /opt/intel/openvino/runtime/cmake)
    find_package(OpenVINO REQUIRED)

    set(SOURCE_FILES openvino/modelManager.cpp)
    set(INCLUDE_DIRS ${OpenVINO_INCLUDE_DIR} /opt/intel/openvino/runtime/include)
    set(LINK_LIBRARIES openvino::runtime)
elseif(RECOGNITION_METHOD MATCHES "Template")
    message(STATUS "Using RECOGNITION METHOD Template")
    add_definitions(-DTEMPLATES_PATH="${PROJECT_SOURCE_DIR}/templateMatch/templates/")
    add_definitions(-DTEMPLATE)

    set(SOURCE_FILES templateMatch/modelManager.cpp)
else()
    message(FATAL_ERROR "Not Define RECOGNITION METHOD")
endif()

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${INCLUDE_DIRS}
)

add_library(DigitalRecognitionLib STATIC ${SOURCE_FILES} InferResult.cpp)
target_link_libraries(DigitalRecognitionLib ${LINK_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark DigitalRecognitionLib ${OpenCV_LIBRARIES})
