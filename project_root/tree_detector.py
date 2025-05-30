# tree_detector.py
from ultralytics import YOLO
import cv2

class TreeDetector:
    def __init__(self, model_path="best.pt"):
        """
        Initializes the YOLOv8 model for tree detection.

        Args:
            model_path (str): Path to the trained YOLO model file.
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def detect_trees(self, frame, imgsz=320, conf_threshold=0.25):
        """
        Detects trees in a given frame.

        Args:
            frame (numpy.ndarray): The input image/frame.
            imgsz (int): Image size for YOLO processing.
            conf_threshold (float): Confidence threshold for detections.

        Returns:
            list: A list of detections. Each detection is a dictionary:
                  {'bbox': (x1, y1, x2, y2), 'confidence': conf, 'class_id': cls}
        """
        if self.model is None:
            return []

        results = self.model(frame, imgsz=imgsz, verbose=False) # verbose=False to reduce console output
        
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]) 
                        # Assuming class_id for tree is known, or only one class is detected
                    })
        return detections

if __name__ == '__main__':
    # Example Usage (requires an image and your model)
    detector = TreeDetector(model_path="best.pt") # Ensure 'best.pt' is accessible
    
    # Create a dummy frame for testing if you don't have an image handy
    # For a real test, load an image:
    # test_image_path = r"C:\Users\kaniu\OneDrive - UGent\UGent\2024-2025\Semester 2\Computervisie\Labo\dataset\raw\eastbound\frame_0188.jpg"
    # frame = cv2.imread(test_image_path)
    
    # Dummy frame for syntax check
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8) 

    if frame is not None and detector.model is not None:
        detected_objects = detector.detect_trees(frame)
        if detected_objects:
            print(f"Detected {len(detected_objects)} trees.")
            for obj in detected_objects:
                print(f"  BBox: {obj['bbox']}, Confidence: {obj['confidence']:.2f}")
                # You can draw these on the frame for visualization
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow("Detections", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("No trees detected in the test image.")
    elif detector.model is None:
        print("YOLO model not loaded, cannot run detection test.")
    else:
        print("Failed to load test image.")