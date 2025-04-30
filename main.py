import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Must be first line!

# Now import other packages
from ultralytics import YOLO
import torch

# Absolute path to YAML (verified working)
DATA_YAML = r'C:\Users\kaniu\OneDrive - UGent\UGent\2024-2025\Semester 2\Computervisie\Labo\cv_based_tree_measurement\Tree Project Kani\dataset\data.yaml'

def train_model():
    # Load model (adjust based on your preference)
    model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy
    
    # Train with CPU optimization
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        batch=4,  # Reduced for CPU
        imgsz=640,
        device='cpu',
        workers=4,  # Adjust based on your CPU cores
        single_cls=True,  # Since you only have 'Trunk' class
        verbose=True
    )
    return results

if __name__ == "__main__":
    train_model()