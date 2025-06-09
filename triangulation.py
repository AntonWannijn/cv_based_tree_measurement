import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
import calc_width as cw

# Add project_root to the Python path
project_root = os.path.join(os.path.dirname(__file__), 'project_root')
sys.path.append(project_root)

import tree_tracker as tt

def detect_objects(model, image, imgsz=320):
    """
    Detect objects in an image using a YOLO model.
    
    Args:
        model (YOLO): YOLO model to use for detection.
        image (ndarray): Image to detect objects in.
        imgsz (int, optional): Image size for processing. Defaults to 320.
        
    Returns:
        list: Detection results.
    """
    return model(image, imgsz=imgsz, verbose = False)

def process_detections(image, model, visualize=True):
    """
    Process all detections in an image.
    
    Args:
        image (ndarray): Image to process.
        model (YOLO): YOLO model to use for detection.
        visualize (bool, optional): Whether to visualize results. Defaults to True.
        
    Returns:
        list: List of dictionaries containing detection results.
    """
    results = detect_objects(model, image)
    processed_results = []
    
    for detection in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = float(detection.conf[0])
        class_id = int(detection.cls[0])
        
        # Extract and process ROI
        roi = cw.extract_roi(image, (x1, y1, x2, y2))
        mask = cw.preprocess_roi(roi)
        
        """
        if visualize:
            cw.display_image("Blurred", mask)
        """
        
        # Find edge distance
        left_point, right_point, distance = cw.find_edge_distance(mask)
        
        """
        # Add visualization if needed
        if visualize:
            mask_with_line = cw.visualize_edges(mask, left_point, right_point)
            cw.display_image("Edges", mask_with_line)
        """
        
        # Store result
        processed_results.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'class_id': class_id,
            'left_edge': left_point,
            'right_edge': right_point,
            'edge_distance': distance
        })
        
        #print(f"Detection {len(processed_results)}: Distance between two vertical edges: {distance} pixels")
    
    return processed_results

def main():
    """
    Main function to demonstrate usage of all functions.
    """
    # Configuration
    image_paths = []
    model_path = r"C:\Users\Hannes\Documents\GitHub\cv_based_tree_measurement\best.pt"
    model = cw.load_model(model_path)
    folder = r"C:\Users\Hannes\Documents\School\5de_jaar\Computervisie\frames\eastbound_20240319"

    for file in os.listdir(folder):
        if file.lower().endswith('.jpg'):
            full_path = os.path.join(folder, file)
            image_paths.append(full_path)

    # Load model and image
    images = []
    for image_path in image_paths:
        images.append(cw.load_image(image_path))

    # Process all detections
    results = []
    for i in range(10):
        image = images[i]
        results.append(process_detections(image, model))

    tracker = tt.BasicTracker()
    for result in results:
        tracker.update(result)
    
    for track_id in tracker.tracks:
        print(f"Track {track_id} info:", tracker.tracks[track_id])
    
    """
    # Print summary
    for result in results:
        print(f"Found {len(result)} detections")
        for i, result in enumerate(result):
            print(f"Detection {i+1}: Edge distance = {result['edge_distance']} pixels")
    """

if __name__ == "__main__":
    main()
