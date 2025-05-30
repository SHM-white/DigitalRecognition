function(detectBackend)
    if (${NN_BACKEND} MATCHES "auto")
        find_package(OpenVINO)
        find_package(CUDA)
        if(EXISTS "/usr/local/include/rknn_api.h")
            set(_NN_BACKEND "rknn")
        elseif(${CUDA_FOUND})
            set(_NN_BACKEND "tensorrt")
        elseif(${OpenVINO_FOUND})
            set(_NN_BACKEND "openvino")
        else()
            message(FATAL_ERROR "No Support NN Backend Found")
        endif()
    else()
        set(_NN_BACKEND ${NN_BACKEND})
    endif()
    if (${RECOGNITION_METHOD} MATCHES "Template")
        add_definitions(-DTEMPLATE)
    elseif (${_NN_BACKEND} MATCHES "tensorrt")
        find_package(CUDA)
        add_definitions(-DTENSORRT)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            add_definitions(-DJETSON)
            include_directories(/usr/local/cuda/targets/aarch64-linux/include)
            link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
        else ()
            include_directories(${CUDA_INCLUDE_DIRS})
            include_directories(${CUDA_}/include)
            include_directories(${TensorRT_DIR_}/include)
            link_directories(${CUDA_}/lib64)
            link_directories(${TensorRT_DIR_}/lib)
        endif ()
    elseif(${_NN_BACKEND} MATCHES "openvino")
        find_package(OpenVINO)
        add_definitions(-DOPENVINO)
        include_directories(${OpenVINO_INCLUDE_DIR} /opt/intel/openvino/runtime/include)
    elseif(${_NN_BACKEND} MATCHES "rknn")
        add_definitions(-DRKNN)
        message(DEBUG "Using RKNN")
    else()
        message(FATAL_ERROR "Not Support NN Backend")
    endif()
endfunction(detectBackend)

function(detectMethod)
    if(${RECOGNITION_METHOD} MATCHES "MobileNet")
        add_definitions(-DMOBILE_NET)
    elseif(RECOGNITION_METHOD MATCHES "BP")
        add_definitions(-DBP)
    elseif(RECOGNITION_METHOD MATCHES "Template")
        add_definitions(-DTEMPLATE)
    else()
        message(FATAL_ERROR "Not Define RECOGNITION METHOD")
    endif()
endfunction(detectMethod)
