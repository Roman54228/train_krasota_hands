#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Camera - Обработка видео с камеры в реальном времени

Приложение для детекции рук и keypoints с использованием TensorRT моделей
с поддержкой depth данных от ToF камеры.

Использование:
    python process_camera_new.py

Возможности:
    - Детекция рук с помощью YOLO
    - Предсказание keypoints и классификация жестов
    - 3D координаты с использованием depth данных
    - OSC отправка результатов
    - Визуализация в реальном времени

Автор: Refactored version
Дата: 2025
"""

import time
import numpy as np
import cv2
import torch
from collections import defaultdict
import os

# Импорт внешних модулей
from coordinate_transformer import CoordinateTransformer
from osc_sender import Sender
from tensorrt_utils import load_hand_gesture_model, run_model_batch

# Try to import pyvidu for camera access
try:
    import pyvidu as vidu
    PYVIDU_AVAILABLE = True
except ImportError:
    print("pyvidu module not found. Camera functionality will be limited.")
    PYVIDU_AVAILABLE = False


class CameraProcessor:
    """Основной класс для обработки видео с камеры"""
    
    def __init__(self, yolo_model_path: str, trt_model_path: str, kps_model_path: str, 
                 osc_ip: str = "10.0.0.101", osc_port: int = 5055):
        # Инициализация координатного трансформера
        self.coord_transformer = CoordinateTransformer()
        
        # Инициализация OSC отправителя
        self.sender = Sender(ip=osc_ip, port=osc_port, logging_level="DEBUG")
        
        # Загрузка моделей
        print("Загрузка YOLO модели...")
        from ultralytics import YOLO
        self.model = YOLO(yolo_model_path)
        
        print("Загрузка TensorRT моделей...")
        self.trt_model = load_hand_gesture_model(trt_model_path)
        self.kps_model = load_hand_gesture_model(kps_model_path)
        
        # Создание словаря для хранения предыдущих детекций каждого трека
        self.previous_detections = defaultdict(list)
        
        # Параметры обработки
        self.CROP_SIZE = 256
        self.map_colors = {0: (255,0,0), 1: (255,128,0), 2: (102,255,255), 3: (153,0,153), 4: (178,102,55)}
        
        # YOLO классификация рук
        self.yolo_classes = {1: 'hand_l', 0: 'hand_r', 2: 'twohands'}
        
        # ToF параметры
        self.TOF_PARAMS = {
            "ToF::StreamFps": 5,
            "ToF::Distance": 7.5,
            "ToF::Exposure": 5.6,
            "ToF::DepthMedianBlur": 0,
            "ToF::DepthFlyingPixelRemoval": 1,
            "ToF::Threshold": 100,
            "ToF::Gain": 2,
            "ToF::AutoExposure": 0,
            "ToF::DepthSmoothStrength": 0,
            "ToF::DepthCompletion": 0,
        }
        
        # Параметры depth
        self.DEPTH_SCALE_ALPHA = 2000.0 / 60000.0
        self.RAW_DEPTH_MULTIPLIER = 2
        self.MAX_DEPTH_VALUE = 65535.0
        self.DEFAULT_TOF_RANGE_M = 7.5
        
        print("Инициализация завершена")
    
    def smooth_detection(self, track_id, current_detection):
        """Сглаживание детекций"""
        # Добавляем текущую детекцию для этого трека
        self.previous_detections[track_id].append(current_detection)

        # Ограничиваем количество сохранённых детекций
        if len(self.previous_detections[track_id]) > 2:
            self.previous_detections[track_id].pop(0)

        # Сглаживание: берём среднее значение последних 2 детекций
        smoothed_box = np.mean(self.previous_detections[track_id], axis=0).astype(int)
        return smoothed_box
    
    def extract_camera_intrinsics(self, stream):
        """Извлечение параметров камеры"""
        if not PYVIDU_AVAILABLE:
            print("Warning: pyvidu not available, using default camera parameters")
            return False
            
        intrinsics = vidu.intrinsics()
        extrinsics = vidu.extrinsics()
        
        if not stream.getCamPara(intrinsics, extrinsics):
            return False
        
        # Extract all non-private, non-callable attributes
        attrs = {
            name.lower(): getattr(intrinsics, name) 
            for name in dir(intrinsics)
            if not name.startswith('__') and not callable(getattr(intrinsics, name))
        }
        
        # Try to get focal lengths and principal point
        fx = attrs.get('fx', attrs.get('f_x'))
        fy = attrs.get('fy', attrs.get('f_y'))
        cx = attrs.get('cx', attrs.get('c_x'))
        cy = attrs.get('cy', attrs.get('c_y'))
        
        # Fallback to intrinsic matrix K if individual parameters not available
        if (fx is None or fy is None or cx is None or cy is None) and hasattr(intrinsics, 'K'):
            K_matrix = np.array(getattr(intrinsics, 'K'), dtype=float).reshape(3, 3)
            fx, fy, cx, cy = K_matrix[0, 0], K_matrix[1, 1], K_matrix[0, 2], K_matrix[1, 2]
        
        if fx is None or fy is None or cx is None or cy is None:
            return False
        
        # Convert to float
        fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
        
        # Set camera matrix
        self.coord_transformer.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Extract distortion coefficients
        k1 = float(attrs.get('k1', 0.0))
        k2 = float(attrs.get('k2', 0.0))
        p1 = float(attrs.get('p1', 0.0))
        p2 = float(attrs.get('p2', 0.0))
        k3 = float(attrs.get('k3', 0.0))
        
        self.coord_transformer.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        
        print(f"Camera intrinsics updated: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        return True
    
    def configure_tof_stream(self, stream):
        """Настройка параметров ToF потока"""
        if not PYVIDU_AVAILABLE:
            return
            
        try:
            for param_name, param_value in self.TOF_PARAMS.items():
                stream.set(param_name, param_value)
        except Exception as e:
            print(f"Warning: Failed to set some ToF parameters: {e}")
    
    def calculate_depth_distance(self, raw_depth: float) -> float:
        """Преобразование raw depth в расстояние в метрах"""
        if raw_depth <= 0:
            return 0.0
        raw_scaled = raw_depth * self.RAW_DEPTH_MULTIPLIER
        return raw_scaled * (self.DEFAULT_TOF_RANGE_M / self.MAX_DEPTH_VALUE)
    
    def process_frame(self, image, depth=None):
        """Обработка одного кадра"""
        # Обработка изображения (если оно в оттенках серого - конвертируем в BGR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        draw_image = image.copy()
        image_h, image_w = image.shape[:2]
        
        # Инференс YOLO
        results = self.model.predict(image)
        
        list_src_crops, resized_crops = [], []
        detections = []
        
        batch_size = 1
        # Обработка детекций
        for i, box in enumerate(results[0].boxes.xyxy[:batch_size]):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(results[0].boxes.cls[i].item())
            yolo_confidence = float(results[0].boxes.conf[i].item())
            
            # Получаем название класса YOLO
            yolo_class_name = self.yolo_classes.get(cls, f'unknown_{cls}')
            
            cropped_hand = image[y1:y2, x1:x2]
            list_src_crops.append(cropped_hand)
            cropped_resized = cv2.resize(cropped_hand, (self.CROP_SIZE, self.CROP_SIZE))
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
                self.sender.send(address=f"/bboxes/bbox_{i}/point_{j}", data=[pt[0], pt[1], pt[2]])
            
            # Отправка YOLO классификации
            self.sender.send(address=f"/bboxes/bbox_{i}/yolo_class", data=[cls, yolo_confidence, yolo_class_name])
            
            track_id = 0
        
        # Если нет детекций, возвращаем оригинальное изображение
        if len(list_src_crops) == 0:
            return draw_image, []
        
        # Предсказание keypoints и классификация
        with torch.no_grad():
            output = run_model_batch(resized_crops, self.kps_model, image_size=256)
        kps_preds = output[0]
        
        cls_output = run_model_batch(resized_crops, self.trt_model, image_size=256)[0]
        
        # Отрисовка результатов
        for i, box in enumerate(results[0].boxes.xyxy[:batch_size]):
            x1, y1, x2, y2 = map(int, box.tolist())
            yolo_cls = int(results[0].boxes.cls[i].item())
            yolo_conf = float(results[0].boxes.conf[i].item())
            yolo_class_name = self.yolo_classes.get(yolo_cls, f'unknown_{yolo_cls}')
            
            label = cls_output[i].argmax()
            confidence = float(cls_output[i].max())
            color = self.map_colors[label]
            
            preds = np.expand_dims(kps_preds[i], 0)[:, :, :2] * 256
            preds = preds[0]
            h, w, _ = list_src_crops[i].shape
            points = []
            
            # Подготовка keypoints для отправки (21 точка с 3D координатами)
            keypoints_3d = np.zeros((21, 3))
            
            for p_id, (x_norm, y_norm) in enumerate(preds):
                x_norm, y_norm = x_norm / self.CROP_SIZE, y_norm / self.CROP_SIZE
                px = int(x_norm * w)
                py = int(y_norm * h)
                abs_px = x1 + px
                abs_py = y1 + py
                
                # Получаем глубину для keypoint
                z = 0
                if depth is not None and 0 <= abs_py < depth.shape[0] and 0 <= abs_px < depth.shape[1]:
                    raw_depth = depth[abs_py, abs_px]
                    z = self.calculate_depth_distance(raw_depth)
                
                # Сохраняем 3D координаты keypoint
                keypoints_3d[p_id] = [abs_px, abs_py, z]
                
                if p_id == 0:
                    my_x, my_y = abs_px, abs_py
                    cv2.circle(draw_image, (my_x, my_y), 3, (0, 255, 0), -1)
                    
                    # Координаты в мировой системе (если есть depth)
                    if z > 0:
                        real_x, real_y, real_z = self.coord_transformer.pixel_to_floor_3d(my_x, my_y, z)
                        cv2.putText(draw_image, f'{real_x:.1f} {real_y:.1f} {real_z:.2f}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                points.append((abs_px, abs_py))
                cv2.circle(draw_image, (abs_px, abs_py), 2, (0, 255, 0), -1)
            
            # Отправка результатов через OSC
            self.sender.send_hand_detection(
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
        
        return draw_image, list_src_crops
    
    def find_tof_stream(self, device):
        """Поиск ToF потока"""
        if not PYVIDU_AVAILABLE:
            return None
            
        num_streams = device.getStreamNum()
        
        for stream_idx in range(num_streams):
            with vidu.PDstream(device, stream_idx) as stream:
                if not stream.init():
                    continue
                
                stream_name = stream.getStreamName().upper()
                if "TOF" in stream_name:
                    return stream_idx
        
        return None
    
    def run_camera_processing(self):
        """Основной цикл обработки видео с камеры"""
        if not PYVIDU_AVAILABLE:
            print("Error: pyvidu not available. Cannot access camera.")
            return
        
        # Инициализация устройства
        device = vidu.PDdevice()
        if not device.init():
            print("Error: Device initialization failed")
            return
        
        # Поиск ToF потока
        tof_stream_idx = self.find_tof_stream(device)
        if tof_stream_idx is None:
            print("Error: No ToF stream found")
            return
        
        # Настройка OpenCV окна
        cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
        
        # Открытие и настройка ToF потока
        with vidu.PDstream(device, tof_stream_idx) as tof_stream:
            if not tof_stream.init():
                print("Error: ToF stream initialization failed")
                return
            
            # Настройка параметров потока
            self.configure_tof_stream(tof_stream)
            
            # Извлечение параметров камеры
            if not self.extract_camera_intrinsics(tof_stream):
                print("Warning: Could not extract camera intrinsics")
            
            print("Нажмите 'q' для выхода")
            
            # Основной цикл обработки
            while True:
                # Получение кадров из ToF потока
                images = tof_stream.getPyImage()
                
                # Проверка валидности кадров
                if not images or len(images) < 2 or images[1] is None or images[1].size == 0:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Обработка кадров
                ir_frame = images[1]
                depth_frame = images[0]
                ir_processed = cv2.flip(ir_frame, -1)
                depth_processed = cv2.flip(depth_frame, -1)
                
                # Обработка кадра
                processed_image, crops = self.process_frame(ir_processed, depth_processed)
                
                # Отображение результата
                cv2.imshow("Hand Detection", processed_image)
                
                # Проверка на выход
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Очистка
        cv2.destroyAllWindows()
        print("Обработка завершена")


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обработка видео с камеры для детекции рук')
    parser.add_argument('--yolo', type=str, default="/home/cineai/ViduSdk/python/TRT_Roma/newone/upside.engine",
                        help='Путь к YOLO модели')
    parser.add_argument('--trt', type=str, default="my_model.engine",
                        help='Путь к TensorRT модели классификации')
    parser.add_argument('--kps', type=str, default="cls_kps.engine",
                        help='Путь к TensorRT модели keypoints')
    parser.add_argument('--osc-ip', type=str, default="10.0.0.101",
                        help='IP адрес для OSC отправки')
    parser.add_argument('--osc-port', type=int, default=5055,
                        help='Порт для OSC отправки')
    
    args = parser.parse_args()
    
    # Создание процессора
    processor = CameraProcessor(
        yolo_model_path=args.yolo,
        trt_model_path=args.trt,
        kps_model_path=args.kps,
        osc_ip=args.osc_ip,
        osc_port=args.osc_port
    )
    
    # Запуск обработки
    processor.run_camera_processing()


if __name__ == "__main__":
    main()
