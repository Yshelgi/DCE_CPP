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
    int index = 0;
    const unsigned char* imgData = img.ptr<unsigned char>(0);
    for (int h = 0; h < img_h; h++){
        index = h * img_w;
        for (int w = 0; w < img_w; w++){
            data[index + w] = static_cast<float>(imgData[index * 3 + 3 * w])/255.;
            data[step + index + w] = static_cast<float>(imgData[index * 3 + 3 * w + 1])/255.;
            data[step * 2 + index + w] = static_cast<float>(imgData[index * 3 + 3 * w + 2])/255.;
        }
    }
}


void TRTInfer::preprocess(const cv::Mat& image, const cv::Mat& nchw,float* data){
    this->pparam.height = image.rows;
    this->pparam.width  = image.cols;

    blobFromImage(nchw,data); //speed up
    //this->context->setInputShape("images", nvinfer1::Dims{4, {1, 3, size.height,size.width}});
    SetCudaDevice(this->deviceId);
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], data, nchw.total() * 3 * sizeof(float), cudaMemcpyHostToDevice, this->stream));
    // delete[] data;
    // data = nullptr;
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




