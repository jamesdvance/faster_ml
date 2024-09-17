from yolov8 import YOLOv8

model_path = "models/teeth_yolov8.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

