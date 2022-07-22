#pragma once
#include <opencv2/opencv.hpp>

float* get_images_slicing(const std::vector<cv::Mat>& lst_face_align, int lower, int upper, int input_width, int input_height)
{

    float* image_data;
    int batch_size = upper - lower;
    int each_image_size = input_width * input_height * 3;
    float* data = new float[batch_size * each_image_size];
    // Starting and Ending iterators
    for (int i = 0; i < batch_size; i++){
        cv::Mat img = lst_face_align[lower + i];
        for (int j = 0; j < input_width * input_height; j++) {
            // Transpose image
            data[i*each_image_size + j]                                  = ((float)img.at<cv::Vec3b>(j)[0]/255.0 - 0.5)/0.5;
            data[i*each_image_size + j + input_width * input_height]     = ((float)img.at<cv::Vec3b>(j)[1]/255.0 - 0.5)/0.5;
            data[i*each_image_size + j + 2 * input_width * input_height] = ((float)img.at<cv::Vec3b>(j)[2]/255.0 - 0.5)/0.5;
        }
    }

    // Return the final sliced vector
    return data;
}



void normalize_vector(std::shared_ptr<float[]> arr, int features_size, std::shared_ptr<float[]> norm_features){
    float sum_square = 0;
    for (int i = 0; i < features_size; i++){
        sum_square += arr[i]*arr[i];
    }
    sum_square = std::sqrt(sum_square);
    for (int i = 0; i < features_size; i++){
        norm_features[i] = arr[i]/sum_square;
    }
}   