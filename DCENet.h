#pragma once
#include "infer.h"


class DCENet : public TRTInfer{
public:
    DCENet(const std::string& enginePath,int deviceId=0);
    virtual ~DCENet() = default;
    void postprocess(cv::Mat &image) override;
    void run(const cv::Mat& img,cv::Mat& dst,cv::Size size=cv::Size(640,640));
};


