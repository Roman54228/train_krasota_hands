import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO

model_yolo_path = "my_experiments/run18/weights/best.pt"
input_image_folder = "/home/ladmin/Desktop/hand_rec/frames_okulo_distorted_15082025/"
basedir = os.path.basename(input_image_folder)
# output_images_dir = "roma_images/"
image_files = glob(os.path.join(input_image_folder, "*"))
# image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

yolo_model = YOLO(model_yolo_path)
CROP_SIZE = 400
j = 0
# os.makedirs(f'krasota_pinch_crops/{basedir}/')
for image_path in tqdm(image_files):
    j += 1
    # if j < 950:
    #     continue
    # if j % 4 != 0:
    #     continue
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    image = cv2.imread(image_path)
    # cv2.imwrite(f'full_frames/{bsn_src_dir}/{filename}', image)
    original_image = image
    vis_image = original_image.copy()
    
    image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0) / 255
    if image is None:
        continue

    # Инференс YOLO
    results = yolo_model(original_image)
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = int(results[0].boxes.cls[i].item())

        box_width = x2 - x1
        box_height = y2 - y1
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        
        expand_width = int(0.01 * box_width)
        expand_height = int(0.01 * box_height)

        x1 = max(0, x1 - expand_width)
        y1 = max(0, y1 - expand_height)
        x2 += expand_width
        y2 += expand_height
        
        crop = original_image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (0,0), fx=3, fy=3)
        # breakpoint()
        cv2.imwrite(f'crops_frames_okulo_distorted_15082025/{i}_{filename}', crop)