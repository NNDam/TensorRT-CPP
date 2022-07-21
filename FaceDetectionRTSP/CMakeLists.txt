cmake_minimum_required(VERSION 2.6)

project(face_detection)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda-11.4/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda-11.4/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda-11.4/include)
    link_directories(/usr/local/cuda-11.4/lib64)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")



find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/*.cpp
)

add_executable(face_detectors ${SOURCES})
target_link_libraries(face_detectors nvinfer)
target_link_libraries(face_detectors cudart)
target_link_libraries(face_detectors ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
