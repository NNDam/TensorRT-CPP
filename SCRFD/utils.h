#pragma once
#include <opencv2/opencv.hpp>

// Return <batch, scale>
void get_images_slicing(std::vector<cv::Mat> lst_rgb_image,
                                            int lower,
                                            int upper,
                                            int input_width,
                                            int input_height,
                                            float* tmp_input,
                                            float* tmp_scales)
{
    int img_w, img_h;
    int batch_size = upper - lower;
    int each_image_size = input_width * input_height * 3;
    cv::Mat img_resized;
    // Starting and Ending iterators
    for (int i = 0; i < batch_size; i++){
        cv::Mat img = lst_rgb_image[lower + i];
        
        // Resize keep ratio
        img_w = img.size().width;
        img_h = img.size().height;
        float scale = std::min((float)input_width/(float)img_w, (float)input_height/(float)img_h);
        
        int new_img_w = (int)(img_w * scale);
        int new_img_h = (int)(img_h * scale);

        std::cout << "From " << img_w << " x " << img_h << " resize to " << new_img_w << " x " << new_img_h << std::endl;
        cv::resize(img, img_resized, cv::Size(new_img_w, new_img_h));

        // Padding & Transpose image
        tmp_scales[i] = scale;
        for (int j = 0; j < input_height; j++) {
            for (int k = 0; k < input_width; k++){
                int pixel_index = j*input_height + k;
                if ((j < new_img_h) && (k < new_img_w)){
                    // 3 channels
                    tmp_input[i*each_image_size + 0 + pixel_index]                          = ((float)img_resized.at<cv::Vec3b>(j, k)[0] - 127.5)/128.0;
                    tmp_input[i*each_image_size + input_width*input_height + pixel_index]   = ((float)img_resized.at<cv::Vec3b>(j, k)[1] - 127.5)/128.0;
                    tmp_input[i*each_image_size + 2*input_width*input_height + pixel_index] = ((float)img_resized.at<cv::Vec3b>(j, k)[2] - 127.5)/128.0;
                }
                else{
                    tmp_input[i*each_image_size + pixel_index]                              = -127.5/128.0;
                    tmp_input[i*each_image_size + input_width*input_height + pixel_index]   = -127.5/128.0;
                    tmp_input[i*each_image_size + 2*input_width*input_height + pixel_index] = -127.5/128.0;
                }

            }
            
        }
    }
}
