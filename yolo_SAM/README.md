# Install
TO Start with, please download the pre-trained weights for SAM at https://github.com/facebookresearch/segment-anything, There are three pre-trained weights of different parameter sizes available for download, namely ViT-H SAM model, ViT-L SAM model, and ViT-B SAM model.

Then, downloading the pre-trained weights for yolov8 refers to https://docs.ultralytics.com/zh/models/yolov8

# Usage
1. First， you should train (or fine-tuning) a yolov8 model to detect the location of organoids.
2. then using the box as prompt to SAM.
