import cv2
import numpy as np

image = cv2.imread('dataset/train/images/frame_0001.jpg')
height, width = image.shape[:2]

with open('dataset/train/labels/frame_0001.txt', 'r') as f:
    for line in f.readlines():
        cls, x_center, y_center, w, h = map(float, line.split())
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Label Check', image)
cv2.waitKey(0)