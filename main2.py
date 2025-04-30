from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # or yolov8s/m/l/x for larger models

model.train(data="trunk_dataset.yaml", imgsz=480)  # Set input size to 480x480

model.train(
    data="trunk_dataset.yaml",
    epochs=30,
    batch=32,
    imgsz=480,
    lr0=0.0001,
    momentum=0.9,
    weight_decay=0.0005,
    pretrained=True,  # Start from COCO weights
    device=0,         # Use GPU
)

for param in model.model.backbone.parameters():
    param.requires_grad = False