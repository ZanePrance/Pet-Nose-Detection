import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Train the pet nose localizer.')

parser.add_argument('-dataset', '--dataset_path', type=str, help='Path to test images')
args = parser.parse_args()

dataset_path = args.dataset_path

model = YOLO('yolov8n.pt')

results = model.train(data=dataset_path, epochs=100, imgsz=640)
