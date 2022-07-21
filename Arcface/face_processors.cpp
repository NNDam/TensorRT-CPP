#include <iostream>
#include <chrono>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h" //Allocate, deallocate CUDA memory
#include "NvInfer.h"
#include "face_processors.h"
#include "utils.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


// Overwrite
FaceProcessor::FaceProcessor(int max_batch_size, int input_width, int input_height, int features_size) :
        max_batch_size_(max_batch_size),
        input_width_(input_width),
        input_height_(input_height),
        features_size_(features_size)
{
}

FaceProcessor::~FaceProcessor()
{
}

void FaceProcessor::init()
{
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    assert(mRuntime != nullptr);

    std::cout << "Deserialize Engine" << std::endl;
    deserializeEngine();

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    assert(mContext != nullptr);

    mContext->setOptimizationProfile(0);

    std::cout << "Finished init" << std::endl;
}

void FaceProcessor::deserializeEngine(){
    std::ifstream file("../weights/model.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char* trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        mCudaEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream, size));
        assert(mCudaEngine != nullptr);
    }
    else{
        throw std::invalid_argument("Read weight file error");
    }
}

void FaceProcessor::inferenceOnce(nvinfer1::IExecutionContext& context, float* input, float* output, int batch_size){
    // Get engine
    const nvinfer1::ICudaEngine& engine = context.getEngine();

    // Get index of input & output in engine
    const int input_index  = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int output_index = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Set context binding dimension
    context.setBindingDimensions(input_index, nvinfer1::Dims4(batch_size, 3, input_height_, input_width_));

    nvinfer1::Dims tmp = context.getBindingDimensions(input_index);
    // Pointers to input and output DEVICE memories/buffers
    void* buffers[2]; // just 1 input & 1 output

    // Allocate Input & Output in DEVICE memory
    int size_of_input  = batch_size * input_width_ * input_height_ * 3 * sizeof(float);
    int size_of_output = batch_size * features_size_ * sizeof(float);
    CHECK(cudaMalloc(&buffers[input_index], size_of_input));
    CHECK(cudaMalloc(&buffers[output_index], size_of_output));

    // Create Stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Copy Memory from Host to Device (Async)
    CHECK(cudaMemcpyAsync(buffers[input_index], input, size_of_input, cudaMemcpyHostToDevice, stream));

    // Inference (Async)
    // For Async: enqueue() & enqueueV2()
    // For Sync:  execute() & executeV2()
    // enqueue() & execute() will be deprecated in TensorRT 8.4
    // std::cout << "enqueueV2 " << std::endl;
    context.enqueueV2(buffers, stream, nullptr);

    // Copy Memory from Device to Host (Async)
    CHECK(cudaMemcpyAsync(output, buffers[output_index], size_of_output, cudaMemcpyDeviceToHost, stream));
    
    // Wait for all concurrent executions in stream to be completed
    cudaStreamSynchronize(stream);

    // Deallocate
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[input_index]));
    CHECK(cudaFree(buffers[output_index]));
}

void FaceProcessor::get_features(const std::vector<cv::Mat>& lst_face_align, std::vector <float*>& lst_features){
    // Get number of batch
    int total_image = lst_face_align.size();
    int total_minibatch = total_image % max_batch_size_ == 0 ? (int)(total_image / max_batch_size_) : (int)(total_image / max_batch_size_) + 1;
    int lower;
    int upper;
    int batch_size;
    float* tmp_output    = new float[max_batch_size_ * features_size_]; // Must allocate memory
    float* tmp_features  = new float[features_size_]; // Allocate once
    float* tmp_input;
    // Infer each minibatch
    for (int i = 0; i < total_minibatch; i++){

        lower = i*max_batch_size_;
        upper = min((i+1)*max_batch_size_, total_image);
        batch_size = upper - lower;

        tmp_input = get_images_slicing(lst_face_align, lower, upper, input_width_, input_height_);

        inferenceOnce(*mContext, tmp_input, tmp_output, batch_size);

        // Parse output features
        for (int j = 0; j < batch_size; j++){
            for (int k = 0; k < features_size_; k++){
                tmp_features[k] = tmp_output[j*features_size_ + k];
            }
            float* norm_features = new float[features_size_]; // Allocate once
            normalize_vector(tmp_features, features_size_, norm_features);
            
            lst_features.push_back(norm_features);
        }
    }
    delete[] tmp_input;
    delete[] tmp_output;
    delete[] tmp_features;
}

int main(){
    cudaSetDevice(0);
    // Read & resize image
    std::cout << "Preprocess image" << std::endl;
    std::vector<cv::Mat> lst_face_align;
    cv::Mat img = cv::imread("../test_images/00011.jpg");
    cv::resize(img, img, cv::Size(112, 112));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // std::cout << "img = " << std::endl << " "  << img << std::endl << std::endl;
    lst_face_align.push_back(img);
    cv::Mat img2 = cv::imread("../test_images/crop.jpg");
    cv::resize(img2, img2, cv::Size(112, 112));
    cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    // std::cout << "img2 = " << std::endl << " "  << img2 << std::endl << std::endl;
    lst_face_align.push_back(img2);

    // Init model & Load weights
    std::cout << "Init model" << std::endl;
    FaceProcessor face_embedding(1, 112, 112, 512);
    face_embedding.init();
    
    // Inference
    std::cout << "Infer" << std::endl;
    auto start = std::chrono::system_clock::now();
    std::vector <float*> lst_features;
    face_embedding.get_features(lst_face_align, lst_features);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for(float* feature: lst_features){
        for (int j = 0; j < 10; j++){
            std::cout << feature[j] << " ";
        }
        std::cout << std::endl;
    }

    
}