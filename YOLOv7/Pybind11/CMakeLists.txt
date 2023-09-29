cmake_minimum_required(VERSION 3.4)

project(object_detection)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -I/usr/include/python3.8 -lpython3.8")

find_package(Python COMPONENTS Interpreter Development)
message("-- Python_FOUND:${Python_FOUND}")
message("-- Python_VERSION:${Python_VERSION}")
message("-- Python_Development_FOUND:${Python_Development_FOUND}")
message("-- Python_LIBRARIES:${Python_LIBRARIES}")


find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(Threads REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package( OpenCV REQUIRED )
message("-- OpenCV_FOUND='${OpenCV_FOUND}'")
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/yolov7)


# pybind11 method:
pybind11_add_module(object_detection main.cpp yolov7/utils.cpp yolov7/yolov7.cpp)


# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer)
target_link_libraries(${PROJECT_NAME} PRIVATE ${pybind11_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} PRIVATE cudart)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_THREAD_LIBS_INIT})