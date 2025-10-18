#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Images - Обработка изображений с детекцией рук

Приложение для обработки изображений с детекцией рук и keypoints
с использованием TensorRT моделей и 3D координат.

Использование:
    python process_images_new.py --input /path/to/images --output /path/to/results

Возможности:
    - Детекция рук с помощью YOLO
    - Предсказание keypoints и классификация жестов
    - 3D координаты с использованием depth данных
    - OSC отправка результатов
    - Обработка папок с изображениями

Автор: Refactored version
Дата: 2025
"""

from ultralytics import YOLO
import time
import numpy as np
import cv2
import torch
from collections import defaultdict
import os
import glob
from pathlib import Path

# Импорт внешних модулей
from coordinate_transformer import CoordinateTransformer
from osc_sender import Sender
from tensorrt_utils import load_hand_gesture_model, run_model_batch

# Инициализация координатного трансформера
coord_transformer = CoordinateTransformer()

# Инициализация OSC отправителя (опционально)
sender = Sender(ip="10.0.0.101", port=5055, logging_level="DEBUG")

# Загрузка моделей
model = YOLO("/home/cineai/ViduSdk/python/TRT_Roma/newone/upside.engine")
trt_model = load_hand_gesture_model("my_model.engine")
kps_model = load_hand_gesture_model("cls_kps.engine")

# Создание словаря для хранения предыдущих детекций каждого трека
previous_detections = defaultdict(list)

# Функция для сглаживания детекций
def smooth_detection(track_id, current_detection):
    # Добавляем текущую детекцию для этого трека
    previous_detections[track_id].append(current_detection)

    # Ограничиваем количество сохранённых детекций
    if len(previous_detections[track_id]) > 2:
        previous_detections[track_id].pop(0)

    # Сглаживание: берём среднее значение последних 2 детекций
    smoothed_box = np.mean(previous_detections[track_id], axis=0).astype(int)
    return smoothed_box

# Функция для обработки изображений из папки
def process_images_from_folder(input_folder, output_folder=None, depth_folder=None):
    """
    Обрабатывает изображения из папки
    
    Args:
        input_folder: путь к папке с входными изображениями
        output_folder: путь к папке для сохранения результатов (если None, создается папка 'results')
        depth_folder: путь к папке с depth картами (опционально)
    """
    # Создание папки для результатов
    if output_folder is None:
        output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    
    # Получение списка изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"Не найдено изображений в папке: {input_folder}")
        return
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    print(f"Результаты будут сохранены в: {output_folder}")
    
    CROP_SIZE = 256
    map_colors = {0: (255,0,0), 1: (255,128,0), 2: (102,255,255), 3: (153,0,153), 4: (178,102,55)}
    
    # YOLO классификация рук
    yolo_classes = {1: 'hand_l', 0: 'hand_r', 2: 'twohands'}
    
    # Обработка каждого изображения
    for img_idx, image_path in enumerate(image_files):
        print(f"Обработка {img_idx + 1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка чтения изображения: {image_path}")
            continue
        
        # Чтение depth карты если указана папка
        depth = None
        if depth_folder:
            depth_path = os.path.join(depth_folder, os.path.basename(image_path))
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Обработка изображения (если оно в оттенках серого - конвертируем в BGR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        draw_image = image.copy()
        image_h, image_w = image.shape[:2]
        
        # Инференс YOLO
        results = model.predict(image)
        
        list_src_crops, resized_crops = [], []
        detections = []
        
        batch_size = 1
        # Обработка детекций
        for i, box in enumerate(results[0].boxes.xyxy[:batch_size]):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(results[0].boxes.cls[i].item())
            yolo_confidence = float(results[0].boxes.conf[i].item())
            
            # Получаем название класса YOLO
            yolo_class_name = yolo_classes.get(cls, f'unknown_{cls}')
            
            cropped_hand = image[y1:y2, x1:x2]
            list_src_crops.append(cropped_hand)
            cropped_resized = cv2.resize(cropped_hand, (CROP_SIZE, CROP_SIZE))
            resized_crops.append(cropped_resized)
            
            # Сохраняем текущую детекцию
            current_detection = [x1, y1, x2, y2]
            
            # Расширяем бокс на 1%
            x1_smooth, y1_smooth, x2_smooth, y2_smooth = x1, y1, x2, y2
            expand_width = int(0.01 * (x2_smooth - x1_smooth))
            expand_height = int(0.01 * (y2_smooth - y1_smooth))
            
            x1_smooth = max(0, x1_smooth - expand_width)
            y1_smooth = max(0, y1_smooth - expand_height)
            x2_smooth = min(image_w, x2_smooth + expand_width)
            y2_smooth = min(image_h, y2_smooth + expand_height)
            
            z = float(0)
            pt_tl = [float(x1_smooth), float(y1_smooth), z]
            pt_tr = [float(x2_smooth), float(y1_smooth), z]
            pt_bl = [float(x1_smooth), float(y2_smooth), z]
            pt_br = [float(x2_smooth), float(y2_smooth), z]
            
            vctl = np.array(pt_tl)
            vcbr = np.array(pt_br)
            pt_ct = (vcbr + vctl) / 2
            
            pts = (pt_ct, pt_tl, pt_tr, pt_bl, pt_br)
            for j, pt in enumerate(pts):
                sender.send(address=f"/bboxes/bbox_{i}/point_{j}", data=[pt[0], pt[1], pt[2]])
            
            # Отправка YOLO классификации
            sender.send(address=f"/bboxes/bbox_{i}/yolo_class", data=[cls, yolo_confidence, yolo_class_name])
            
            track_id = 0
        
        # Если нет детекций, пропускаем
        if len(list_src_crops) == 0:
            print(f"  Не найдено рук на изображении")
            # Сохраняем оригинальное изображение
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, draw_image)
            continue
        
        # Предсказание keypoints и классификация
        with torch.no_grad():
            output = run_model_batch(resized_crops, kps_model, image_size=256)
        kps_preds = output[0]
        
        cls_output = run_model_batch(resized_crops, trt_model, image_size=256)[0]
        
        # Отрисовка результатов
        for i, box in enumerate(results[0].boxes.xyxy[:batch_size]):
            x1, y1, x2, y2 = map(int, box.tolist())
            yolo_cls = int(results[0].boxes.cls[i].item())
            yolo_conf = float(results[0].boxes.conf[i].item())
            yolo_class_name = yolo_classes.get(yolo_cls, f'unknown_{yolo_cls}')
            
            label = cls_output[i].argmax()
            confidence = float(cls_output[i].max())
            color = map_colors[label]
            
            preds = np.expand_dims(kps_preds[i], 0)[:, :, :2] * 256
            preds = preds[0]
            h, w, _ = list_src_crops[i].shape
            points = []
            
            # Подготовка keypoints для отправки (21 точка с 3D координатами)
            keypoints_3d = np.zeros((21, 3))
            
            for p_id, (x_norm, y_norm) in enumerate(preds):
                x_norm, y_norm = x_norm / CROP_SIZE, y_norm / CROP_SIZE
                px = int(x_norm * w)
                py = int(y_norm * h)
                abs_px = x1 + px
                abs_py = y1 + py
                
                # Получаем глубину для keypoint
                z = 0
                if depth is not None and 0 <= abs_py < depth.shape[0] and 0 <= abs_px < depth.shape[1]:
                    z = depth[abs_py, abs_px] * (7.5/65536)
                
                # Сохраняем 3D координаты keypoint
                keypoints_3d[p_id] = [abs_px, abs_py, z]
                
                if p_id == 0:
                    my_x, my_y = abs_px, abs_py
                    cv2.circle(draw_image, (my_x, my_y), 3, (0, 255, 0), -1)
                    
                    # Координаты в мировой системе (если есть depth)
                    if z > 0:
                        real_x, real_y, real_z = coord_transformer.pixel_to_floor_3d(my_x, my_y, z)
                        cv2.putText(draw_image, f'{real_x:.1f} {real_y:.1f} {real_z:.2f}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                points.append((abs_px, abs_py))
                cv2.circle(draw_image, (abs_px, abs_py), 2, (0, 255, 0), -1)
            
            # Отправка результатов через OSC
            sender.send_hand_detection(
                bbox_id=i,
                gesture_class=int(label),
                confidence=confidence,
                keypoints=keypoints_3d
            )
            
            # Рисуем соединения между keypoints
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (5,6),(6,7),(7,8),
                (9,10),(10,11),(11,12),
                (13,14),(14,15),(15,16),
                (17,18),(18,19),(19,20),
                (0,5),(5,9),(9,13),(13,17),(0,17)
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(points) and end_idx < len(points):
                    start = points[start_idx]
                    end = points[end_idx]
                    cv2.line(draw_image, start, end, (0, 255, 0), thickness=2)
            
            # Рисуем bounding box
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 2)
            
            # Отображаем информацию о YOLO классификации и жесте
            cv2.putText(draw_image, f'YOLO: {yolo_class_name} ({yolo_conf:.2f})', (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(draw_image, f'Gesture: {label} ({confidence:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Сохранение результата
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, draw_image)
        print(f"  Сохранено: {output_path}")

# Пример использования
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Обработка изображений с детекцией рук')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Путь к папке с входными изображениями')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Путь к папке для сохранения результатов (по умолчанию: results)')
    parser.add_argument('--depth', '-d', type=str, default=None,
                        help='Путь к папке с depth картами (опционально)')
    
    args = parser.parse_args()
    
    process_images_from_folder(args.input, args.output, args.depth)
