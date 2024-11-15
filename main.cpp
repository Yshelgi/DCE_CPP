#include <iostream>
#include <fstream>
#include <filesystem>
#include <dirent.h>
#include "DCENet.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

void mkdir(std::string dirPath) {
    if(!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directory(dirPath);
    }
}


int main() {
    mkdir("./result");
    std::string engineFile = "/home/xtxk/Desktop/python_code/Zero-DCE_extension/Zero-DCE++/DCE.engine";
    std::string dirPath = "/home/xtxk/Desktop/python_code/Zero-DCE_extension/Zero-DCE++/data/test_data/difficult/*";
    int deviceId = 1;

    auto model = new DCENet(engineFile, deviceId);
    model->make_pipe(true);
    std::vector<std::string> imgLists;
    cv::glob(dirPath, imgLists);
    cv::Mat img,dst;
    int count = 0;
    for(const auto& file : imgLists) {
        img = cv::imread(file);
        auto st = std::chrono::high_resolution_clock::now();
        model->run(img,dst);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
        std::cout <<"cost: "<< duration.count() << "ms \n";
        cv::imwrite("./result/"+std::to_string(count)+".jpg",dst);
        count++;
    }
}
