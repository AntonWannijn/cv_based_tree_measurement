import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from ultralytics import YOLO 

# Function for stereo vision and depth estimation
import triangulation as tri
import Calibratie as cali
import detect_video as det

camera = cv2.VideoCapture(2)  # Open OBS camera 

# Stereo vision setup parameters
frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 30              # Distance between the cameras [cm]
f = 8               # Camera lens's focal length [mm]
alpha = 49.9        # Camera field of view in the horizontal plane [degrees]

while camera.isOpened():
    ret1, frame1 = camera.read()
    if not ret1:
        print("Error: Could not read frame.")
        break

    # Perform object detection using YOLOv11
    results = det.model(frame1, imgsz=320)  # Smaller imgsz for faster processing
    annotated_frame1 = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Live Tree Detection", annotated_frame1)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break