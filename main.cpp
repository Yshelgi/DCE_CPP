#include <iostream>
#include <fstream>
#include <filesystem>
#include "DCENet.h"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void mkdir(std::string dirPath) {
    if(!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directory(dirPath);
    }
}


int main() {
    mkdir("../result");
    std::string engineFile = "../ckpts/DCE_plus_fp16.engine";
    std::string dirPath = "../test_imgs/*";
    int deviceId = 0;
    cv::Size size(512,512);

    auto model = new DCENet(engineFile, deviceId);
    model->make_pipe(true);
    std::vector<std::string> imgLists;
    cv::glob(dirPath, imgLists);
    cv::Mat img,dst;
    int count = 0;
    for(const auto& file : imgLists) {
        img = cv::imread(file);
        auto st = std::chrono::high_resolution_clock::now();
        model->run(img,dst,size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
        std::cout <<"cost: "<< duration.count() << "ms \n";
        cv::imwrite("../result/"+std::to_string(count)+".jpg",dst);
        count++;
        //break;
    }
}
