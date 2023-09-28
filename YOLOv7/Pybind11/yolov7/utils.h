#include <opencv2/opencv.hpp>

void get_images_slicing(const std::vector<cv::Mat>& lst_rgb_image,
                        int lower,
                        int upper,
                        int input_width,
                        int input_height,
                        float* tmp_input,
                        float* tmp_scales);