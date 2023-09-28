#include <stdio.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <opencv2/opencv.hpp>
#include "yolov7/yolov7.h"

int main(int argc, char** argv){

    // Init model
    YOLOV7 object_detector(1, 640, 640, 200);
    object_detector.init();
    
    // Reader & Writer
    std::string rtsp_path = argv[1];
    std::cout << "Read RTSP stream: " << rtsp_path << std::endl;
    cv::VideoCapture cap(rtsp_path);
    cv::VideoWriter writer("output.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 25, cv::Size(640, 480));

    // Inference
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    float threshold = 0.65;
    int count_frame = 0;
    while (1){
        cv::Mat frame;
        cv::Mat frame_bgr;
        // Read frame
        cap >> frame;
        if (frame.empty()) {
            break;
        };
        frame_bgr = frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        
        std::vector<std::vector<ObjectInfo>> lst_objects {};
        std::vector<cv::Mat> lst_rgb_image {};

        
        lst_rgb_image.push_back(frame);

        auto start = std::chrono::system_clock::now();
        object_detector.detect(lst_rgb_image, threshold, lst_objects);
        auto end = std::chrono::system_clock::now();
        float FPS = 1000.0 / (float) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Frame " << std::to_string(count_frame) << " Detected " << lst_objects[0].size() << " object(s) | FPS: " << FPS << std::endl;

        object_detector.visualize_all(frame_bgr, lst_objects[0]);
        cv::resize(frame_bgr, frame_bgr, cv::Size(640, 480));
        writer.write(frame_bgr);

        lst_objects.clear();
        lst_rgb_image.clear();
        count_frame += 1;
    };
    cap.release();
    writer.release();
    return 0;
}