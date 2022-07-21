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
	int max_batch_size_ = 4;
    int input_width_    = 112;
    int input_height_   = 112;
    int keep_top_k_     = 200;
    const char* INPUT_BLOB_NAME = "input.1";
    const char* OUTPUT_BLOB_NAME_NUM_DETECTIONS = "num_detections";
    const char* OUTPUT_BLOB_NAME_BOXES = "nmsed_boxes";
    const char* OUTPUT_BLOB_NAME_SCORES = "nmsed_scores";
    const char* OUTPUT_BLOB_NAME_CLASSES = "nmsed_classes";
    const char* OUTPUT_BLOB_NAME_LANDMARKS = "nmsed_landmarks";
};
