# Python & Pybind11 C++ wrapper for End-to-End YOLOv7
- Original repo: https://github.com/WongKinYiu/yolov7
- NMS Plugin: https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin
## 1. Tested Performance
Tested performance with ```NVIDIA A100-PCIE-40GB - 1x3x640x640```
|    Model   |   FP32  |   FP16  |   Download  |
|:----------:|:-------:|:-------:|:-------:|
|   YOLOv7-NMS   |   6.14 ms  |   3.79 ms  |   [download](https://drive.google.com/file/d/1BrB4mAX71pA5cl3_NgCazgDF5WP9xUYh/view?usp=sharing)  |
|   YOLOv7X-NMS  |   7.96 ms  |   4.51 ms  |   [download](https://drive.google.com/file/d/1c4gp_m_u0Zoo7N8mIQsErwnPxVkU4-bJ/view?usp=sharing)  |
|  YOLOv7-W6-NMS |   6.30 ms  |   4.28 ms  |   [download](https://drive.google.com/file/d/1lth50WR-71SKfDUTW9cMTezF0iDOoKAx/view?usp=sharing)  |
|  YOLOv7-E6-NMS |   8.12 ms  |   5.18 ms  |   [download](https://drive.google.com/file/d/1PQj4iKjhNjd5kz6BXHhqp6HhfCnbjeqk/view?usp=sharing)  |
|  YOLOv7-D6-NMS |   9.29 ms  |   5.79 ms  |   [download](https://drive.google.com/file/d/1R_Gc8NtBPXdiAnP0qsSFxRv__FikNJe0/view?usp=sharing)  |
|  YOLOv7-E6E-NMS |   10.82 ms  |   6.88 ms  |   [download](https://drive.google.com/file/d/1au_ZplYP2m2JkayIuDE1ygMNOaV5zjFe/view?usp=sharing)  |

## 2. TensorRT Python
### Test Environment
- torch 1.11.0
- onnx 1.12.0

### Convert to ONNX
You can get all original MSCOCO converted ONNX models [here](https://drive.google.com/drive/folders/15hUBefQv28FJ-yfw_Wpvlbu23WeY5E2S?usp=sharing)
or convert model yourself:
- Clone and put ```tools/export.py``` and ```tools/add_nms_plugins.py``` to [original yolov7 repo](https://github.com/WongKinYiu/yolov7)
- Run ```python export.py --weights yolov7x.pt --dynamic  --simplify``` to get ```yolov7x.onnx```
- Run ```python add_nms_plugins.py --model yolov7x.onnx``` to add NMS Plugin and get ```yolov7x-nms.onnx```

<i> Note: This is dynamic shape ONNX model </i>
### Convert to TensorRT
Convert ONNX to TensortRT with dynamic input shape in range ```1x3x640x640```-```4x3x896x896```
```
  /usr/src/tensorrt/bin/trtexec --onnx=yolov7x-nms.onnx \
                              --saveEngine=yolov7x-nms-fp16.trt \
                              --explicitBatch \
                              --minShapes=input:1x3x640x640 \
                              --optShapes=input:1x3x640x640 \
                              --maxShapes=input:4x3x896x896 \
                              --verbose \
                              --device=1 \
                              --workspace=512 \
                              --fp16
```
#### Run Sample Demo
Change path to serialized engine file and run 
```
  python object_detector_trt_nms.py
```
<p align="center">
  <img src="Python/output/result-640.jpg" width="960"> <br>
  <i> Output of <b>YOLOv7X-1x3x640x640 thresh 0.5</b>. For better result, just resize input larger</i>
</p>

## 3. Pybind11 TensorRT C++
### Sample Pybind11 Application for Video/RTSP input
#### Requirements
- Python3.8 (for other version please edit ```CMakeLists.txt``` manually)
- pybind11: ```pip install pybind11[global]```
- opencv, openmp

Put converted tensorrt model to ```weights/yolov7-nms-fp32.trt```
```
  cd Pybind11
  mkdir build && cd build
  cmake .. && make
  cd ..
  python test_pybind11.py
```
