#include <iostream>
#include <chrono>
#include <stdexcept>
#include <assert.h>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h" //Allocate, deallocate CUDA memory
#include "NvInfer.h"
#include "yolov7.h"
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

static bool GOT_CLASSES_NAME = false;
std::vector<std::string> DICT_CLASSES_NAME;

// Overwrite
YOLOV7::YOLOV7(int max_batch_size, int input_width, int input_height, int keep_top_k) :
        max_batch_size_(max_batch_size),
        input_width_(input_width),
        input_height_(input_height),
        keep_top_k_(keep_top_k)
{
}

YOLOV7::~YOLOV7()
{
    std::cout << "Deallocate Stream & CUDA memory" << std::endl;
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[index_input]));
    CHECK(cudaFree(buffers[index_num_detections]));
    CHECK(cudaFree(buffers[index_nmsed_boxes]));
    CHECK(cudaFree(buffers[index_nmsed_scores]));
    CHECK(cudaFree(buffers[index_nmsed_classes]));
    
}

void YOLOV7::init()
{
    // Load classes name
    std::cout << "Prepare classes name " << std::endl;
    std::ifstream fdict;
    setlocale(LC_CTYPE, "");
    if(!GOT_CLASSES_NAME) {
        fdict.open(classes_name);
        if(!fdict.is_open())
        {
            std::cout << "open dictionary file failed." << std::endl;
            abort();
        }
        while(!fdict.eof()) {
            std::string strLineAnsi;
            if (getline(fdict, strLineAnsi) ) {
                // std::cout << strLineAnsi << std::endl;
                // if (strLineAnsi.length() > 1) {
                //     strLineAnsi.erase(1);
                strLineAnsi.erase(strLineAnsi.size() - 1);
                // }
                // std::cout << strLineAnsi << std::endl;
                DICT_CLASSES_NAME.push_back(strLineAnsi);
            }
        }
        GOT_CLASSES_NAME=true;
        fdict.close();
    }
    initLibNvInferPlugins(&gLogger, "");
    
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    assert(mRuntime != nullptr);

    std::cout << "Deserialize Engine" << std::endl;
    deserializeEngine();
    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    assert(mContext != nullptr);
    mContext->setOptimizationProfile(0);

    // Create Stream
    std::cout << "Create Stream" << std::endl;
    CHECK(cudaStreamCreate(&stream));

    std::cout << "Allocate CUDA once time for reusing" << std::endl;
    // Allocate Input & Output in DEVICE memory
    const nvinfer1::ICudaEngine& engine = mContext->getEngine();
    index_input           = engine.getBindingIndex(INPUT_BLOB_NAME);
    index_num_detections  = engine.getBindingIndex(OUTPUT_BLOB_NAME_NUM_DETECTIONS);
    index_nmsed_boxes     = engine.getBindingIndex(OUTPUT_BLOB_NAME_BOXES);
    index_nmsed_scores    = engine.getBindingIndex(OUTPUT_BLOB_NAME_SCORES);
    index_nmsed_classes   = engine.getBindingIndex(OUTPUT_BLOB_NAME_CLASSES);
    assert (index_input >= 0);
    assert (index_nmsed_classes >= 0);
    assert (index_num_detections >= 0);
    assert (index_nmsed_boxes >= 0);
    assert (index_nmsed_scores >= 0);
    int size_of_input           = max_batch_size_ * input_width_ * input_height_ * 3 * sizeof(float);
    int size_of_num_detections  = max_batch_size_ * sizeof(int);
    int size_of_nmsed_boxes     = max_batch_size_ * keep_top_k_ * 4 * sizeof(float);
    int size_of_nmsed_scores    = max_batch_size_ * keep_top_k_ * sizeof(float);
    int size_of_nmsed_classes   = max_batch_size_ * keep_top_k_ * sizeof(float);
    CHECK(cudaMalloc(&buffers[index_input], size_of_input));
    CHECK(cudaMalloc(&buffers[index_num_detections], size_of_num_detections));
    CHECK(cudaMalloc(&buffers[index_nmsed_boxes], size_of_nmsed_boxes));
    CHECK(cudaMalloc(&buffers[index_nmsed_scores], size_of_nmsed_scores));
    CHECK(cudaMalloc(&buffers[index_nmsed_classes], size_of_nmsed_classes));
    std::cout << "Finished init" << std::endl;
}

void YOLOV7::deserializeEngine(){
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

void YOLOV7::inferenceOnce(nvinfer1::IExecutionContext& context,
                                    float* input,
                                    int* num_detections,
                                    float* nmsed_boxes,
                                    float* nmsed_scores,
                                    float* nmsed_classes,
                                    int batch_size){

    // Set context binding dimension
    context.setBindingDimensions(index_input, nvinfer1::Dims4(batch_size, 3, input_height_, input_width_));

    // Copy Memory from Host to Device (Async)
    int size_of_input           = batch_size * input_width_ * input_height_ * 3 * sizeof(float);
    int size_of_num_detections  = batch_size * sizeof(int);
    int size_of_nmsed_boxes     = batch_size * keep_top_k_ * 4 * sizeof(float);
    int size_of_nmsed_scores    = batch_size * keep_top_k_ * sizeof(float);
    int size_of_nmsed_classes   = batch_size * keep_top_k_ * sizeof(float);
    CHECK(cudaMemcpyAsync(buffers[index_input], input, size_of_input, cudaMemcpyHostToDevice, stream));

    // Inference (Async)
    // For Async: enqueue() & enqueueV2()
    // For Sync:  execute() & executeV2()
    // enqueue() & execute() will be deprecated in TensorRT 8.4
    // std::cout << "enqueueV2 " << std::endl;
    context.enqueueV2(buffers, stream, nullptr);

    // Copy Memory from Device to Host (Async)
    CHECK(cudaMemcpyAsync(num_detections, buffers[index_num_detections], size_of_num_detections, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(nmsed_boxes, buffers[index_nmsed_boxes], size_of_nmsed_boxes, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(nmsed_scores, buffers[index_nmsed_scores], size_of_nmsed_scores, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(nmsed_classes, buffers[index_nmsed_classes], size_of_nmsed_classes, cudaMemcpyDeviceToHost, stream));
    
    // Wait for all concurrent executions in stream to be completed
    cudaStreamSynchronize(stream);
      
}


void YOLOV7::visualize_all(cv::Mat& visualize_image,
                                std::vector<ObjectInfo>& lst_objects){
    for (ObjectInfo obj: lst_objects){
        int class_id      = obj.classId;
        float class_prob  = obj.confidence;
        std::string label = DICT_CLASSES_NAME[class_id] + " (" + std::to_string(class_prob) + ")"; 
        cv::Point pt1((int)obj.x1, (int)obj.y1);
        cv::Point pt2((int)obj.x2, (int)obj.y2);
        cv::Point pt3((int)obj.x1, (int)obj.y1 - 10);

        cv::rectangle(visualize_image, pt1, pt2, cv::Scalar(0, 255, 0), 4);
        cv::putText(visualize_image, label, pt3, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, false);

    }
    
}

void YOLOV7::detect(const std::vector<cv::Mat>& lst_rgb_image,
                            const float& threshold,
                            std::vector<std::vector<ObjectInfo>>& lst_objects){
    // Get number of batch
    int total_image = lst_rgb_image.size();
    int total_minibatch = total_image % max_batch_size_ == 0 ? (int)(total_image / max_batch_size_) : (int)(total_image / max_batch_size_) + 1;
    int lower;
    int upper;
    int batch_size;
    int each_image_size = input_width_ * input_height_ * 3;

    float* tmp_input             = new float[max_batch_size_ * each_image_size];
    float* tmp_scales            = new float[max_batch_size_];
    int* tmp_num_detections      = new int[max_batch_size_ * 1]; // Must allocate memory
    float* tmp_nmsed_boxes       = new float[max_batch_size_ * keep_top_k_ * 4]; // Must allocate memory
    float* tmp_nmsed_scores      = new float[max_batch_size_ * keep_top_k_ * 1]; // Must allocate memory
    float* tmp_nmsed_classes     = new float[max_batch_size_ * keep_top_k_ * 1]; // Must allocate memory
    
    // Infer each minibatch
    for (int i = 0; i < total_minibatch; i++){
        lower = i*max_batch_size_;
        upper = min((i+1)*max_batch_size_, total_image);
        batch_size = upper - lower;
        // Preprocess
        get_images_slicing(lst_rgb_image, lower, upper, input_width_, input_height_, tmp_input, tmp_scales);
        // Inference
        inferenceOnce(*mContext,
                    tmp_input,
                    tmp_num_detections,
                    tmp_nmsed_boxes,
                    tmp_nmsed_scores,
                    tmp_nmsed_classes,
                    batch_size);
        // Post Process
        for (int j = 0; j < batch_size; j++){
            // Foreach image
            std::vector<ObjectInfo> objs {};
            
            int nbox = tmp_num_detections[j];

            for (int k = 0; k < nbox; ++k){
                if (tmp_nmsed_scores[j*keep_top_k_ + k] > threshold){
                    ObjectInfo obj;
                    // Class ID
                    obj.classId = tmp_nmsed_classes[j*keep_top_k_ + k];
                    // Class Probability
                    obj.confidence = tmp_nmsed_scores[j*keep_top_k_ + k];
                    // Box
                    obj.x1 = tmp_nmsed_boxes[(j*keep_top_k_ + k)*4 + 0] / tmp_scales[j];
                    obj.y1 = tmp_nmsed_boxes[(j*keep_top_k_ + k)*4 + 1] / tmp_scales[j];
                    obj.x2 = tmp_nmsed_boxes[(j*keep_top_k_ + k)*4 + 2] / tmp_scales[j];
                    obj.y2 = tmp_nmsed_boxes[(j*keep_top_k_ + k)*4 + 3] / tmp_scales[j];
                    
                    objs.push_back(obj);

                }
                else {
                    break;
                }
            }
            lst_objects.push_back(objs);
        }
    }
    delete[] tmp_input;           
    delete[] tmp_scales;           
    delete[] tmp_num_detections;   
    delete[] tmp_nmsed_boxes;      
    delete[] tmp_nmsed_scores;    
    delete[] tmp_nmsed_classes;
  
}
