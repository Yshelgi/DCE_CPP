#pragma once

#include <memory>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "common.h"

class TRTInfer{
public:
    TRTInfer();
    virtual ~TRTInfer();
    void make_pipe(bool warmup = true);
    virtual void preprocess(const cv::Mat& nchw,float* data);
    virtual void infer();
    virtual void postprocess(cv::Mat& image){};

    int                  num_bindings = 0;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    int                  deviceId = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    PreParam pparam;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine  = nullptr;
    std::shared_ptr<nvinfer1::IRuntime>          runtime = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};