from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="D:\\EDGE MATRIX internship\\Plastic bottle detection\\Plastic Bottle Image Dataset\\data.yaml", epochs=20)