import os
import time
import cv2
import numpy as np
from numpy import random

from exec_backends.trt_loader import TrtModelNMS
# from models.models import Darknet

def resize_add_padding(bgr_img, new_shape):
    h, w, _ = bgr_img.shape
    scale   = min(new_shape / h, new_shape / w)
    inp     = np.zeros((new_shape, new_shape, 3), dtype = np.uint8)
    nw      = int(w * scale)
    nh      = int(h * scale)
    inp[:nh, :nw] = cv2.resize(bgr_img, (nw, nh))
    return inp, scale

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class YOLOV7(object):
    def __init__(self, 
            model_weights = '../C++/weights/model.engine', 
            max_size = 640, 
            names = 'data/coco.names'):
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.max_size = max_size
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def detect(self, bgr_img, visualize_threshold = 0.5):   
        # Prediction
        ## Padded resize
        inp, scale = resize_add_padding(bgr_img, new_shape=self.max_size)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp, 0)
        print(inp.shape)        
        
        ## Inference
        t1 = time.time()
        num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)
        print(num_detection)
        print(nmsed_bboxes)
        print(nmsed_scores)
        t2 = time.time()
        print('Time cost: ', t2 - t1)
        ## Apply NMS
        num_detection = num_detection[0][0]
        nmsed_bboxes  = nmsed_bboxes[0]
        nmsed_scores  = nmsed_scores[0]
        nmsed_classes  = nmsed_classes[0]
        print('Detected {} object(s)'.format(num_detection))
        # Rescale boxes from img_size to im0 size
        _, _, height, width = inp.shape
        h, w, _ = bgr_img.shape
        nmsed_bboxes /= scale
        visualize_img = bgr_img.copy()
        for ix in range(num_detection):       # x1, y1, x2, y2 in pixel format
            if nmsed_scores[ix] < visualize_threshold:
                break
            cls = int(nmsed_classes[ix])
            label = '%s %.2f' % (self.names[cls], nmsed_scores[ix])
            x1, y1, x2, y2 = nmsed_bboxes[ix]

            cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(cls)], 2)
            cv2.putText(visualize_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(cls)], 2, cv2.LINE_AA)

        cv2.imwrite('output/result-640.jpg', visualize_img)
        return visualize_img

if __name__ == '__main__':
    model = YOLOV7()
    img = cv2.imread('test_images/img.png')
    model.detect(img)

