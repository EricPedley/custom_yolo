from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

model.train(
    data='yolo_data_config.yaml',
    epochs=100,
)