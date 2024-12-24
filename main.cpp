#include <iostream>
#include <fstream>
#include <filesystem>
#include "DCENet.h"
#include <opencv2/opencv.hpp>
#include "cmdline.h"

namespace fs = std::filesystem;
cv::Size maxSize(1920, 1080);

std::string enhanceFilePath(const std::string& originalPath) {
    // 检查路径是否有效
    if (!fs::exists(originalPath) || !fs::is_regular_file(originalPath)) {
        throw std::runtime_error("Invalid file path or file does not exist.");
    }

    // 获取文件的路径和文件名
    fs::path path(originalPath);
    std::string filename = path.filename().string();

    // 找到文件名中最后一个点的位置，用于分割文件名和扩展名
    size_t lastDot = filename.rfind('.');
    if (lastDot == std::string::npos) {
        throw std::runtime_error("File has no extension.");
    }

    // 构造新的文件名
    std::string newFilename = filename.substr(0, lastDot) + "_enhance" + filename.substr(lastDot);
    fs::path newPath = path.parent_path() / newFilename;

    // 返回新的文件路径
    return newPath.string();
}

void enhanceImg(const std::string& imgPath,const std::string& modelPath,int deviceId) {
    cv::Mat img = cv::imread(imgPath);
    cv::Size defaultSize;
    bool isResize = false;
    if (img.cols > 1920 || img.rows > 1080) {
        defaultSize = maxSize;
        isResize = true;
    }else {
        defaultSize = img.size();
    }

    auto model = new DCENet(modelPath, deviceId,defaultSize);
    model->setResize(isResize);
    model->make_pipe(true);

    std::string enhance_imgPath = enhanceFilePath(imgPath);
    cv::Mat dst;
    float* data = new float[defaultSize.width * defaultSize.height * 3];
    model->run(img,dst,data,defaultSize);
    cv::imwrite(enhance_imgPath,dst);
    delete[] data;
    data = nullptr;
}


void enhanceVideo(const std::string& videoPath,const std::string& modelPath,int deviceId) {
    std::string enhance_videoPath = enhanceFilePath(videoPath);
    cv::Mat img,dst;
    cv::VideoCapture cap(videoPath);
    int width,height;
    double fps;
    width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Size defaultSize;
    bool isResize = false;
    if (width > 1920 || height > 1080) {
        defaultSize = maxSize;
        isResize = true;
    }else {
        defaultSize = cv::Size(width,height);
    }

    auto model = new DCENet(modelPath, deviceId,defaultSize);
    model->setResize(isResize);
    model->make_pipe(true);

    float* data = new float[defaultSize.width * defaultSize.height * 3];

    fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(enhance_videoPath,cv::VideoWriter::fourcc('M', 'P', '4', 'V'),fps,cv::Size(width,height));
    while(true) {
        if(!cap.isOpened()) {
            break;
        }
        cap >> img;
        if(img.empty()) {
            break;
        }
        model->run(img,dst,data,defaultSize);
        writer.write(dst);
    }
    cv::destroyAllWindows();
    cap.release();
    writer.release();
    delete[] data;
    data = nullptr;
}

void mkdir(std::string dirPath) {
    if(!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directory(dirPath);
    }
}


int main(int argc,char* argv[]) {
    // 定义cli解析
    cmdline::parser p;

    p.add<std::string>("type",'t',"The type of media type (image/video)",true, "",cmdline::oneof<std::string>("image","video"));
    p.add<std::string>("mediaPath",'p',"The path of media",true,"");
    p.add<std::string>("modelPath",'m',"The path of engine model",false,"./ckpts/DCE_plus3_fp16.engine");
    p.add<int>("deviceId",'d',"Select the GPU ID for model inference",false,0);

    p.parse_check(argc,argv);

    std::string type = p.get<std::string>("type");
    std::string path = p.get<std::string>("mediaPath");
    std::string modelPath = p.get<std::string>("modelPath");
    int deviceId = p.get<int>("deviceId");

    if (type == "image") {
        enhanceImg(path,modelPath,deviceId);
    }else if (type == "video") {
        enhanceVideo(path,modelPath,deviceId);
    }else {
        throw std::runtime_error("Invalid type.");
    }
    return 0;
}
