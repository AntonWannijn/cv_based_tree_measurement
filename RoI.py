import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # Replace with your model path

image_path = r"C:\Users\kaniu\OneDrive - UGent\UGent\2024-2025\Semester 2\Computervisie\Labo\dataset\raw\eastbound\frame_0188.jpg"

image = cv2.imread(image_path)

# Run YOLO detection
results = model(image, imgsz=320)  # Smaller imgsz for faster processing

# Extract RoIs from detections
for detection in results[0].boxes:  # Access bounding boxes
    # Get bounding box coordinates (x1, y1, x2, y2)
    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Convert to integers
    confidence = detection.conf[0]  # Confidence score
    class_id = int(detection.cls[0])  # Class ID

    # Extract the RoI from the image
    roi = image[y1:y2, x1:x2]

    # Apply Canny Edge Detection to the RoI
    edges = cv2.Canny(roi, 100, 200)  # Adjust thresholds as needed

    # Display the edges
    cv2.imshow(f"Edges - Class {class_id}, Conf {confidence:.2f}", edges)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
