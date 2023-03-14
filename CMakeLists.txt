cmake_minimum_required(VERSION 3.10)
project(DigitalRecognition)

set(CMAKE_CXX_STANDARD 14)

set(OpenVINO_DIR /opt/intel/openvino/runtime/cmake)
find_package(OpenVINO REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIR}
        ${OpenVINO_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        /opt/intel/openvino/runtime/include
)

add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/model.xml")

add_library(DigitalRecognitionLib STATIC modelManager.cpp)
target_link_libraries(DigitalRecognitionLib openvino::runtime ${OpenCV_LIBRARIES})

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark DigitalRecognitionLib ${OpenCV_LIBRARIES})
