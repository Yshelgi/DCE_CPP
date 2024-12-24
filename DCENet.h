#pragma once
#include "infer.h"


class DCENet : public TRTInfer{
public:
    DCENet(const std::string &enginePath, int deviceId,const cv::Size& size);
    virtual ~DCENet() = default;
    void postprocess(cv::Mat &image) override;
    void run(const cv::Mat &img, cv::Mat &dst, float* data,cv::Size& size);
    void setResize(bool flag){isResize = flag;}
private:
    bool isResize= false;
};


