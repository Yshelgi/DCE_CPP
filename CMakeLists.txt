cmake_minimum_required(VERSION 3.12)
project(DCE_CPP)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(CMAKE_BUILD_TYPE Release)

# 设置TensorRT编译版本宏定义
#option(USE_V8 "use TensorRT8.x" ON)
#
#if (USE_V8)
#    message("use TensorRT8.x")
#    add_definitions(-DV8)
#else ()
#    message("use TensorRT10.x")
#endif ()

# Add define to track the time consumed at each step.
# you can set it or not
# add_definitions(-DTIMETRACE)

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_LIB_PATH /usr/local/cuda/lib64/)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

set(TensorRT_ROOT /home/xtxk/TensorRT-10.2.0.19)
# set(TensorRT_ROOT /home/TensorRT10)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARIES ${TensorRT_ROOT}/lib)


include_directories(
        .
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)


link_directories(
        ${CUDA_LIB_PATH}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

list(APPEND ALL_LIBS
        ${CUDA_LIB_PATH}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
)

add_executable(DCE_CPP main.cpp DCENet.cpp infer.cpp)
target_link_directories(DCE_CPP PUBLIC ${ALL_LIBS})
target_link_libraries(DCE_CPP PRIVATE nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})
