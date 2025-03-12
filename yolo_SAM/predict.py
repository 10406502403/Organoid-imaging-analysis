import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from ultralytics import YOLO
import os
from SAM import sam_model_registry, SamPredictor
sys.path.append("..")
sam_checkpoint = "sam_b.pt"  #The path of pre-trained SAM model checkpoint 
model_type = "vit_b"  #  the type of vit model ["vit_b", "vit_l","vit_h"], match the sam_checkpoint
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
yolo = YOLO('yolov8_path')  # your trained  YOLOv8 model
img_pth = " your test data path"
save_mask = 'the path to save predicted masks' 
save_box = 'the path to save predicted boxes'
img_list = os.listdir(img_pth)
mask_list = []
for filename in img_list:
    save_dir_mask = os.path.join(save_mask,filename)
    save_dir_box = os.path.join(save_box,filename)
    img_dir = os.path.join(img_pth,filename)
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    results = yolo([img_dir],max_det=1)
    for result in results:
        boxes = result.boxes.xyxy.to(device = predictor.device)
 
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    if transformed_boxes.shape[0]>0:
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        for mask in masks:
            mask = mask.permute(1, 2, 0)  
            mask_output = mask.cpu().numpy()*255
            cv2.imwrite(save_dir_mask,mask_output)
        box = boxes.reshape(4)  
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2) 
        cv2.imwrite(save_dir_box, image) 


    