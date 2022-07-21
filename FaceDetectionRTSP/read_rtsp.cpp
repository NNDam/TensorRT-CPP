#include <stdio.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <opencv2/opencv.hpp>
#include "face_detectors.h"

int main(int argc, char** argv){
    // Init model
    FaceDetectorSCRFD face_detect(1, 640, 640, 200);
    face_detect.init();
    
    // Reader & Writer
    std::string rtsp_path = argv[1];
    std::cout << "Read RTSP stream: " << rtsp_path << std::endl;
    cv::VideoCapture cap(rtsp_path);
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 25, cv::Size(1920, 1080));

    // Inference
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    float threshold = 0.65;
    while (1){
        cv::Mat frame;
        cv::Mat frame_bgr;
        // Read frame
        cap >> frame;
        if (frame.empty()) break;
        frame_bgr = frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        
        std::vector<std::vector<float*>> lst_bboxes {};
        std::vector<std::vector<float*>> lst_points {};
        std::vector<cv::Mat> lst_rgb_image {};

        
        lst_rgb_image.push_back(frame);

        auto start = std::chrono::system_clock::now();
        face_detect.detect(lst_rgb_image, threshold, lst_bboxes, lst_points);
        auto end = std::chrono::system_clock::now();
        float FPS = 1000.0 / (float) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Detected " << lst_bboxes[0].size() << " face(s) | FPS: " << FPS << std::endl;

        face_detect.visualize_all(frame_bgr, lst_bboxes[0], lst_points[0]);
        writer.write(frame_bgr);

    };
    cap.release();
    writer.release();
    return 0;
}