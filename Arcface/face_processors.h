#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "utils.h"
#include "logging.h"

using namespace std;
class FaceProcessor
{
public:
	FaceProcessor(int max_batch_size, int input_width, int input_height, int features_size);
	~FaceProcessor();

	nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
	void deserializeEngine();
	void init();
	void inferenceOnce(nvinfer1::IExecutionContext& context, float* input, float* output, int batch_size);
	void get_features(const std::vector<cv::Mat>& lst_face_align, std::vector <float*>& lst_features);

private:
    Logger gLogger;
	std::shared_ptr<nvinfer1::IRuntime> mRuntime;
	std::shared_ptr<nvinfer1::ICudaEngine> mCudaEngine;
	std::shared_ptr<nvinfer1::IExecutionContext> mContext;
	int max_batch_size_ = 8;
    int input_width_ = 112;
    int input_height_ = 112;
    int features_size_ = 512;
	int input_index = -1;
	int output_index = -1;
    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "output";

	// Pointers to input and output DEVICE memories/buffers
    void* buffers[2]; // just 1 input & 1 output
	cudaStream_t stream;

};
