import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # Replace with your model path

# Open OBS camera
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv11 detection
    results = model(frame, imgsz=320)  # Smaller imgsz for faster processing
    
    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("Live Tree Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()