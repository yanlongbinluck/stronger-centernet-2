
import cv2
import numpy as np
import random
from functools import partial
import torch

def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas

def show_bbox(image_cv2,gt_boxes):
    '''
    image_cv2 numpy [0,255] float32
    '''
    #image = np.copy(image_cv2)
    image = np.ascontiguousarray(image_cv2)
    for i in range(len(gt_boxes)):
        x1,y1,x2,y2 = gt_boxes[i]
        #print((int(x1),int(y1)),(int(x2),int(y2)))
        cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    cv2.imshow('',image/255)
    cv2.imwrite('./test.jpg',image)
    cv2.waitKey(0)

def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                      w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap