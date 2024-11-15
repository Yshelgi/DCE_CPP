#include "DCENet.h"
#include "common.h"

DCENet::DCENet(const std::string &enginePath, int deviceId) {
    SetCudaDevice(deviceId);
    std::ifstream file(enginePath,std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->gLogger));
    assert(this->runtime != nullptr);
    this->deviceId = deviceId;

    this->engine = std::shared_ptr<nvinfer1::ICudaEngine>(this->runtime->deserializeCudaEngine(trtModelStream, size));
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = std::shared_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        auto        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);
        dims = this->engine->getTensorShape(name);
        auto ioType = this->engine->getTensorIOMode(name);
        if(ioType == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name,0,nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            this->context->setInputShape(name, dims);
        }else {
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

void DCENet::postprocess(cv::Mat &image) {
    float* ptr = static_cast<float*>(host_ptrs[0]);
    auto oh = this->output_bindings[0].dims.d[2];
    auto ow = this->output_bindings[0].dims.d[3];
    int dw = pparam.width;
    int dh = pparam.height;
    const int step = ow*oh;

    cv::Mat out = cv::Mat::zeros(ow,oh,CV_8UC3);
    unsigned char* data = out.data;

    // copy data to img && RGB -> BGR
    for(int i=0;i<oh;++i) {
        int index = i*ow;
        for(int j=0;j<ow;++j) {
            data[(i*ow+j)*3] = std::min(255,int(ptr[index + j + 2 * step]*255));
            data[(i*ow+j)*3 + 1] = std::min(255,int(ptr[index + j + step]*255));
            data[(i*ow+j)*3 + 2] = std::min(255,int(ptr[index + j]*255));
        }
    }
    cv::resize(out,image,cv::Size(dw,dh));
}


void DCENet::run(const cv::Mat &img, cv::Mat &dst) {
    cv::Size size(640,640);
    auto st = std::chrono::high_resolution_clock::now();
    preprocess(img,size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"preprocess cost: "<< duration.count() << "ms \n";


    st = std::chrono::high_resolution_clock::now();
    infer();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"infer cost: "<< duration.count() << "ms \n";

    st = std::chrono::high_resolution_clock::now();
    postprocess(dst);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"postprocess cost: "<< duration.count() << "ms \n";
}


