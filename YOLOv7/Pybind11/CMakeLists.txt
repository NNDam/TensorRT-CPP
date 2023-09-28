cmake_minimum_required(VERSION 3.4)

project(object_detection)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package( OpenCV REQUIRED )
message("OpenCV_FOUND='${OpenCV_FOUND}'")
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${PROJECT_SOURCE_DIR}/yolov7)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")



file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/yolov7/*.cpp
    ${PROJECT_SOURCE_DIR}/read_rtsp.cpp
)

add_executable(object_detector ${SOURCES})
target_link_libraries(object_detector nvinfer)
target_link_libraries(object_detector nvinfer_plugin)
target_link_libraries(object_detector cudart)
target_link_libraries(object_detector ${OpenCV_LIBS})
target_link_libraries(object_detector ${CMAKE_THREAD_LIBS_INIT})
