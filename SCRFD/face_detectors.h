#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "utils.h"
#include "logging.h"

using namespace std;
class FaceDetectorSCRFD
{
public:
	FaceDetectorSCRFD(int max_batch_size, int input_width, int input_height, int keep_top_k);
	~FaceDetectorSCRFD();

	nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
	void deserializeEngine();
	void init();
	void inferenceOnce(nvinfer1::IExecutionContext& context,
                        float* input,
                        int* num_detections,
                        float* nmsed_boxes,
                        float* nmsed_scores,
                        float* nmsed_classes,
                        float* nmsed_landmarks,
                        int batch_size);
	void detect(const std::vector<cv::Mat>& lst_rgb_image,
                            const float& threshold,
                            std::vector<std::vector<float*>>& lst_bboxes,
                            std::vector<std::vector<float*>>& lst_points);
    void visualize_all(cv::Mat& visualize_image,
                    std::vector<float*>& bboxes,
                    std::vector<float*>& points);
    void get_face_align(const cv::Mat& rgb_image,
                        const std::vector<float*>& bboxes,
                        const std::vector<float*>& points,
                        std::vector<cv::Mat>& lst_face_align);


private:
    Logger gLogger;
	std::shared_ptr<nvinfer1::IRuntime> mRuntime;
	std::shared_ptr<nvinfer1::ICudaEngine> mCudaEngine;
	std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    // Pointers to input and output DEVICE memories/buffers
    void* buffers[6]; // 1 input & 5 outputs
	int max_batch_size_ = 4;
    int input_width_    = 112;
    int input_height_   = 112;
    int keep_top_k_     = 200;
    int index_input_    = 200;

    int index_input           = -1;
    int index_num_detections  = -1;
    int index_nmsed_boxes     = -1;
    int index_nmsed_scores    = -1;
    int index_nmsed_classes   = -1;
    int index_nmsed_landmarks = -1;

    cudaStream_t stream;

    const char* INPUT_BLOB_NAME = "input.1";
    const char* OUTPUT_BLOB_NAME_NUM_DETECTIONS = "num_detections";
    const char* OUTPUT_BLOB_NAME_BOXES = "nmsed_boxes";
    const char* OUTPUT_BLOB_NAME_SCORES = "nmsed_scores";
    const char* OUTPUT_BLOB_NAME_CLASSES = "nmsed_classes";
    const char* OUTPUT_BLOB_NAME_LANDMARKS = "nmsed_landmarks";
};
