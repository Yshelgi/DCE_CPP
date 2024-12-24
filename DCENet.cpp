#include "DCENet.h"
#include "common.h"

// #define V8

DCENet::DCENet(const std::string &enginePath, int deviceId,const cv::Size& imgSize) {
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

#ifdef V8
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
# else
    this->num_bindings = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        auto        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);
        //dims = this->engine->getTensorShape(name);
        auto ioType = this->engine->getTensorIOMode(name);
        if(ioType == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name,0,nvinfer1::OptProfileSelector::kMAX);
            dims.d[2] = imgSize.width;
            dims.d[3] = imgSize.height;
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
#endif
}

void DCENet::postprocess(cv::Mat &image) {
    float* ptr = static_cast<float*>(host_ptrs[0]);
    auto oh = this->output_bindings[0].dims.d[2];
    auto ow = this->output_bindings[0].dims.d[3];
    // auto oh = int(pparam.height);
    // auto ow = int(pparam.width);
    const int step = ow*oh;

    cv::Mat out = cv::Mat::zeros(ow,oh,CV_8UC3);
    unsigned char* data = out.data;
    int index = 0;
    int tmp = 0;
    // copy data to img && RGB -> BGR
    for(int i=0;i<oh;++i) {
        index = i*ow;
        for(int j=0;j<ow;++j) {
            tmp = (index+j) * 3;
            data[tmp] = std::min(255,int(ptr[index + j + 2 * step]*255));
            data[tmp + 1] = std::min(255,int(ptr[index + j + step]*255));
            data[tmp + 2] = std::min(255,int(ptr[index + j]*255));
        }
    }
    if (this->isResize) {
        cv::resize(out,image,cv::Size(int(pparam.width),int(pparam.height)));
    }else {
        image = std::move(out);
    }
}


void DCENet::run(const cv::Mat &img, cv::Mat &dst, float* data,cv::Size& size) {
#ifdef TIMETRACE
    auto st = std::chrono::high_resolution_clock::now();
#endif
    cv::Mat nchw = img.clone();
    if (isResize) {
        cv::resize(nchw,nchw,size);
    }
    cv::cvtColor(nchw,nchw,cv::COLOR_BGR2RGB);
    preprocess(img,nchw,data);
#ifdef TIMETRACE
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"preprocess cost: "<< duration.count() << "ms \n";
#endif

#ifdef TIMETRACE
    st = std::chrono::high_resolution_clock::now();
#endif
    infer();
#ifdef TIMETRACE
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"infer cost: "<< duration.count() << "ms \n";
#endif

#ifdef TIMETRACE
    st = std::chrono::high_resolution_clock::now();
#endif
    postprocess(dst);
#ifdef TIMETRACE
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
    std::cout <<"postprocess cost: "<< duration.count() << "ms \n";
#endif
}


