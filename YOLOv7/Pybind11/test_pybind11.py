from build.object_detection import YOLOV7

model = YOLOV7(1, 640, 640, 200)
model.init()
model.read_rtsp('test_videos/test_video.mp4', 'output.avi')