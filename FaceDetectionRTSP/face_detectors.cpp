#include <iostream>
#include <chrono>
#include <stdexcept>
#include <assert.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h" //Allocate, deallocate CUDA memory
#include "NvInfer.h"
#include "face_detectors.h"
#include "face_align.h"
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
FaceDetectorSCRFD::FaceDetectorSCRFD(int max_batch_size, int input_width, int input_height, int keep_top_k) :
        max_batch_size_(max_batch_size),
        input_width_(input_width),
        input_height_(input_height),
        keep_top_k_(keep_top_k)
{
}

FaceDetectorSCRFD::~FaceDetectorSCRFD()
{
    std::cout << "Deallocate Stream & CUDA memory" << std::endl;
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[index_input]));
    CHECK(cudaFree(buffers[index_num_detections]));
    CHECK(cudaFree(buffers[index_nmsed_boxes]));
    CHECK(cudaFree(buffers[index_nmsed_scores]));
    CHECK(cudaFree(buffers[index_nmsed_classes]));
    CHECK(cudaFree(buffers[index_nmsed_landmarks]));
    
}

void FaceDetectorSCRFD::init()
{
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
    index_nmsed_landmarks = engine.getBindingIndex(OUTPUT_BLOB_NAME_LANDMARKS);
    assert (index_input >= 0);
    assert (index_nmsed_landmarks >= 0);
    assert (index_nmsed_classes >= 0);
    assert (index_num_detections >= 0);
    assert (index_nmsed_boxes >= 0);
    assert (index_nmsed_scores >= 0);
    int size_of_input           = max_batch_size_ * input_width_ * input_height_ * 3 * sizeof(float);
    int size_of_num_detections  = max_batch_size_ * sizeof(int);
    int size_of_nmsed_boxes     = max_batch_size_ * keep_top_k_ * 4 * sizeof(float);
    int size_of_nmsed_scores    = max_batch_size_ * keep_top_k_ * sizeof(float);
    int size_of_nmsed_classes   = max_batch_size_ * keep_top_k_ * sizeof(float);
    int size_of_nmsed_landmarks = max_batch_size_ * keep_top_k_ * 10 * sizeof(float);
    CHECK(cudaMalloc(&buffers[index_input], size_of_input));
    CHECK(cudaMalloc(&buffers[index_num_detections], size_of_num_detections));
    CHECK(cudaMalloc(&buffers[index_nmsed_boxes], size_of_nmsed_boxes));
    CHECK(cudaMalloc(&buffers[index_nmsed_scores], size_of_nmsed_scores));
    CHECK(cudaMalloc(&buffers[index_nmsed_classes], size_of_nmsed_classes));
    CHECK(cudaMalloc(&buffers[index_nmsed_landmarks], size_of_nmsed_landmarks));
    std::cout << "Finished init" << std::endl;
}

void FaceDetectorSCRFD::deserializeEngine(){
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

void FaceDetectorSCRFD::inferenceOnce(nvinfer1::IExecutionContext& context,
                                    float* input,
                                    int* num_detections,
                                    float* nmsed_boxes,
                                    float* nmsed_scores,
                                    float* nmsed_classes,
                                    float* nmsed_landmarks,
                                    int batch_size){

    // Set context binding dimension
    context.setBindingDimensions(index_input, nvinfer1::Dims4(batch_size, 3, input_height_, input_width_));

    // Copy Memory from Host to Device (Async)
    int size_of_input           = batch_size * input_width_ * input_height_ * 3 * sizeof(float);
    int size_of_num_detections  = batch_size * sizeof(int);
    int size_of_nmsed_boxes     = batch_size * keep_top_k_ * 4 * sizeof(float);
    int size_of_nmsed_scores    = batch_size * keep_top_k_ * sizeof(float);
    int size_of_nmsed_classes   = batch_size * keep_top_k_ * sizeof(float);
    int size_of_nmsed_landmarks = batch_size * keep_top_k_ * 10 * sizeof(float);
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
    CHECK(cudaMemcpyAsync(nmsed_landmarks, buffers[index_nmsed_landmarks], size_of_nmsed_landmarks, cudaMemcpyDeviceToHost, stream));
    
    // Wait for all concurrent executions in stream to be completed
    cudaStreamSynchronize(stream);
      
}


void FaceDetectorSCRFD::visualize_all(cv::Mat& visualize_image,
                                std::vector<std::shared_ptr<float[]>>& bboxes,
                                std::vector<std::shared_ptr<float[]>>& points){
    for (std::shared_ptr<float[]> box: bboxes){
        cv::Point pt1((int)box[0], (int)box[1]);
        cv::Point pt2((int)box[2], (int)box[3]);
        cv::rectangle(visualize_image, pt1, pt2, cv::Scalar(0, 255, 0));
    }
    for (std::shared_ptr<float[]> point: points){
        for (int i = 0; i < 5; i++){
            cv::circle(visualize_image, cv::Point((int)point[i*2], (int)point[i*2+1]), 2, cv::Scalar(0,0,255), cv::FILLED, 2,0);
        }
    }
}

void FaceDetectorSCRFD::get_face_align(const cv::Mat& rgb_image,
                            const std::vector<std::shared_ptr<float[]>>& bboxes,
                            const std::vector<std::shared_ptr<float[]>>& points,
                            std::vector<cv::Mat>& lst_face_align){
    int total_face = bboxes.size();
    cv::Size face_size(112, 112);
    float points_dst[5][2] = {
		{ 30.2946f + 8.0f, 51.6963f },
		{ 65.5318f + 8.0f, 51.5014f },
		{ 48.0252f + 8.0f, 71.7366f },
		{ 33.5493f + 8.0f, 92.3655f },
		{ 62.7299f + 8.0f, 92.2041f }
	};

    for (int i = 0; i < total_face; ++i){
        cv::Mat aligned_face;
        aligned_face.create(face_size, CV_32FC3);
        float points_src[5][2] = {
            {points[i][0], points[i][1]},
            {points[i][2], points[i][3]},
            {points[i][4], points[i][5]},
            {points[i][6], points[i][7]},
            {points[i][8], points[i][9]}
	    };
        cv::Mat src_mat(5, 2, CV_32FC1, points_src);
	    cv::Mat dst_mat(5, 2, CV_32FC1, points_dst);
        cv::Mat transform    = FacePreprocess::SimilarTransform(src_mat, dst_mat);
        cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));
        cv::warpAffine(rgb_image, aligned_face, transfer_mat, face_size, 1, 0, 0);
        lst_face_align.push_back(aligned_face);
    }
}


void FaceDetectorSCRFD::detect(const std::vector<cv::Mat>& lst_rgb_image,
                            const float& threshold,
                            std::vector<std::vector<std::shared_ptr<float[]>>>& lst_bboxes,
                            std::vector<std::vector<std::shared_ptr<float[]>>>& lst_points){
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
    float* tmp_nmsed_landmarks   = new float[max_batch_size_ * keep_top_k_ * 10]; // Must allocate memory
    
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
                    tmp_nmsed_landmarks,
                    batch_size);
        // Post Process
        for (int j = 0; j < batch_size; j++){
            // Foreach image
            std::vector<std::shared_ptr<float[]>> bboxes {};
            std::vector<std::shared_ptr<float[]>> points {};
            
            int nbox = tmp_num_detections[j];

            for (int k = 0; k < nbox; ++k){
                if (tmp_nmsed_scores[j*keep_top_k_ + k] > threshold){
                    std::shared_ptr<float[]> box (new float[4]);
                    std::shared_ptr<float[]> point (new float[10]);
                    for (int ii = 0; ii < 4; ii++){
                        box[ii] = tmp_nmsed_boxes[(j*keep_top_k_ + k)*4 + ii] / tmp_scales[j];
                    }
                    
                    for (int ii = 0; ii < 10; ii++){
                        point[ii] = tmp_nmsed_landmarks[(j*keep_top_k_ + k)*10 + ii] / tmp_scales[j];
                    }
                    bboxes.push_back(box);
                    points.push_back(point);

                }
                else {
                    break;
                }
            }
            lst_bboxes.push_back(bboxes);
            lst_points.push_back(points);

        }
    }
    delete[] tmp_input;           
    delete[] tmp_scales;           
    delete[] tmp_num_detections;   
    delete[] tmp_nmsed_boxes;      
    delete[] tmp_nmsed_scores;    
    delete[] tmp_nmsed_classes; 
    delete[] tmp_nmsed_landmarks;
  
}
