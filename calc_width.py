import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path):
    """
    Load a YOLO model from the specified path.
    
    Args:
        model_path (str): Path to the YOLO model file.
        
    Returns:
        YOLO: Loaded YOLO model.
    """
    return YOLO(model_path)

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        ndarray: Loaded image.
    """
    return cv2.imread(image_path)

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
    return model(image, imgsz=imgsz)

def extract_roi(image, bbox):
    """
    Extract a region of interest from an image based on bounding box coordinates.
    
    Args:
        image (ndarray): Source image.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        
    Returns:
        ndarray: Extracted region of interest.
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def preprocess_roi(roi, blur_kernel_size=9):
    """
    Preprocess a region of interest by applying median blur and Canny edge detection.
    
    Args:
        roi (ndarray): Region of interest to preprocess.
        blur_kernel_size (int, optional): Kernel size for median blur. Defaults to 9.
        
    Returns:
        ndarray: Preprocessed image.
    """
    mask = cv2.medianBlur(roi, blur_kernel_size)
    mask = cv2.Canny(mask, 100, 200)
    return mask

def count_left_edge(row):
    """
    Count pixels from the left until an edge is found.
    
    Args:
        row (ndarray): Row of pixels to analyze.
        
    Returns:
        int: Position of the first edge.
    """
    counter = 0
    for i in range(0, len(row)):
        if row[i] == 255:
            break
        counter += 1
    return counter

def count_right_edge(row):
    """
    Count pixels from the right until an edge is found.
    
    Args:
        row (ndarray): Row of pixels to analyze.
        
    Returns:
        int: Position of the rightmost edge.
    """
    w = len(row) - 1
    counter = w
    for i in range(len(row) - 1, -1, -1):
        if row[i] == 255:
            break
        counter -= 1
    return counter

def draw_line(image, pt1, pt2, color, thickness):
    """
    Draw a line on an image.
    
    Args:
        image (ndarray): Image to draw on.
        pt1 (tuple): First point of the line.
        pt2 (tuple): Second point of the line.
        color (tuple): BGR color of the line.
        thickness (int): Thickness of the line.
        
    Returns:
        ndarray: Image with the line drawn.
    """
    return cv2.line(
        img=image,
        pt1=pt1,
        pt2=pt2,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

def calculate_center(x1, y1, x2, y2):
    """
    Calculate the center point between two points.
    
    Args:
        x1 (int): X-coordinate of the first point.
        y1 (int): Y-coordinate of the first point.
        x2 (int): X-coordinate of the second point.
        y2 (int): Y-coordinate of the second point.
        
    Returns:
        tuple: Center point coordinates.
    """
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def find_edge_distance(mask):
    """
    Find the distance between left and right edges in the middle row of a mask.
    
    Args:
        mask (ndarray): Edge mask image.
        
    Returns:
        tuple: (left_point, right_point, distance)
    """
    h = len(mask) - 1
    w = len(mask[0]) - 1
    
    # Find edges in the middle row
    middle_row = int(h/2)
    left_point = count_left_edge(mask[middle_row])
    right_point = count_right_edge(mask[middle_row])
    
    # Ensure right point is valid
    if right_point == left_point:
        right_point = w
    
    distance = right_point - left_point
    return left_point, right_point, distance

def visualize_edges(mask, left_point, right_point, line_color=(128, 0, 255), thickness=10):
    """
    Visualize edges on a mask by drawing a line between detected edges.
    
    Args:
        mask (ndarray): Edge mask image.
        left_point (int): X-coordinate of the left edge.
        right_point (int): X-coordinate of the right edge.
        line_color (tuple, optional): BGR color of the line. Defaults to (128, 0, 255).
        thickness (int, optional): Thickness of the line. Defaults to 10.
        
    Returns:
        ndarray: Mask with edges visualized.
    """
    h = len(mask) - 1
    middle_row = int(h/2)
    return draw_line(mask, (left_point, middle_row), (right_point, middle_row), line_color, thickness)

def display_image(window_name, image, wait_key=0):
    """
    Display an image in a window.
    
    Args:
        window_name (str): Name of the window.
        image (ndarray): Image to display.
        wait_key (int, optional): Time to wait for a key press (0 = wait indefinitely). Defaults to 0.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()

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
        roi = extract_roi(image, (x1, y1, x2, y2))
        mask = preprocess_roi(roi)
        
        if visualize:
            display_image("Blurred", mask)
        
        # Find edge distance
        left_point, right_point, distance = find_edge_distance(mask)
        
        # Add visualization if needed
        if visualize:
            mask_with_line = visualize_edges(mask, left_point, right_point)
            display_image("Edges", mask_with_line)
        
        # Store result
        processed_results.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'class_id': class_id,
            'left_edge': left_point,
            'right_edge': right_point,
            'edge_distance': distance
        })
        
        print(f"Detection {len(processed_results)}: Distance between two vertical edges: {distance} pixels")
    
    return processed_results

def main():
    """
    Main function to demonstrate usage of all functions.
    """
    # Configuration
    model_path = "best.pt"
    image_path = r"C:\Users\kaniu\OneDrive - UGent\UGent\2024-2025\Semester 2\Computervisie\Labo\dataset\raw\eastbound\frame_0188.jpg"
    
    # Load model and image
    model = load_model(model_path)
    image = load_image(image_path)
    
    # Process all detections
    results = process_detections(image, model)
    
    # Print summary
    print(f"Found {len(results)} detections")
    for i, result in enumerate(results):
        print(f"Detection {i+1}: Edge distance = {result['edge_distance']} pixels")

if __name__ == "__main__":
    main()