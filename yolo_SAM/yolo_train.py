from ultralytics import YOLO
yolo = YOLO('The path to ') 
yolo.train(data='your data cfg path', epochs=100, batch = 4, imgsz=2000, max_det = 1)

