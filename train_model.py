
from ultralytics import YOLO
import os
import torch
import sys
import pickle

cache_data = torch.load('D:\\Project\\Computer Vision\\FlaskApp-main\\dataset\\Data\\Angry.cache')

# Show stats
print(f"Total images cached: {len(cache_data)}")
print(f"Classes found: {set(label for img_path, label in cache_data)}")
print(f"Sample paths: {list(cache_data.keys())[:5]}")