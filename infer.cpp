#include "infer.h"

TRTInfer::TRTInfer()= default;

TRTInfer::~TRTInfer(){
    this->context.reset();
    this->engine.reset();
    this->runtime.reset();
    cudaStreamDestroy(this->stream);
    this->stream = nullptr;
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void TRTInfer::make_pipe(bool warmup)
{
    SetCudaDevice(this->deviceId);
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
    }
}

// devide 255 && BGR -> RGB && HWC -> CHW
void blobFromImage(const cv::Mat& img,float* data) {
    int img_h = img.rows;
    int img_w = img.cols;
    int step = img_h * img_w;
    const unsigned char* imgData = img.ptr<unsigned char>(0);
    for (int h = 0; h < img_h; h++){
        int index = h * img_w;
        for (int w = 0; w < img_w; w++){
            data[index + w] = static_cast<float>(imgData[index * 3 + 3 * w])/255.;
            data[step + index + w] = static_cast<float>(imgData[index * 3 + 3 * w + 1])/255.;
            data[step * 2 + index + w] = static_cast<float>(imgData[index * 3 + 3 * w + 2])/255.;
        }
    }
}


// void blobFromImage(const cv::Mat& img,float* data) {
//     int img_h = img.rows;
//     int img_w = img.cols;
//     int step = img_h * img_w;
//     const unsigned char* imgData = img.ptr<unsigned char>(0);
//     for (int h = 0; h < img_h; h++){
//         int index = h * img_w;
//         for (int w = 0; w < img_w; w++){
//             data[index + w] = static_cast<float>(imgData[index * 3 + 3 * w])/255.;
//             data[index + w + step] = static_cast<float>(imgData[index * 3 + 3 * w + 1])/255.;
//             data[index + w + step * 2] = static_cast<float>(imgData[index * 3 + 3 * w + 2])/255.;
//         }
//     }
// }

void writeFloatArrayToFile(float* array, size_t length, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 保留6位小数
    file << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < length; ++i) {
        file << array[i];
        if (i < length - 1) {
            file << "\t";
        }
        if ((i + 1) % 10 == 0) {
            file << std::endl;
        }
    }
    if ((length + 9) % 10 != 0) {
        file << std::endl;
    }

    file.close();
}


void TRTInfer::preprocess(const cv::Mat& image, cv::Size& size){
    float       height = image.rows;
    float       width  = image.cols;
    cv::Mat nchw,out;
    cv::resize(image, nchw, size);

    // auto st = std::chrono::high_resolution_clock::now();
    // cv::dnn::blobFromImage(nchw, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    // std::cout <<"opencv blob cost: "<< duration.count() << "ms \n";
    // writeFloatArrayToFile(out.ptr<float>(0), nchw.total() * 3, "../blob.csv");

    // st = std::chrono::high_resolution_clock::now();
    cv::cvtColor(nchw, nchw, cv::COLOR_BGR2RGB);
    float* data = new float[nchw.total() * 3];
    blobFromImage(nchw,data); //speed up
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    // std::cout <<"our blob cost: "<< duration.count() << "ms \n";
    // writeFloatArrayToFile(data, nchw.total() * 3, "../myblob.csv");

    this->pparam.height = height;
    this->pparam.width  = width;
    //this->context->setInputShape("images", nvinfer1::Dims{4, {1, 3, size.height,size.width}});
    SetCudaDevice(this->deviceId);
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], data, nchw.total() * 3 * sizeof(float), cudaMemcpyHostToDevice, this->stream));
    delete[] data;
    data = nullptr;
}

void TRTInfer::infer(){
    SetCudaDevice(this->deviceId);
    this->context->executeV2(this->device_ptrs.data());
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}




