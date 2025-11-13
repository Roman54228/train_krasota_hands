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
DEPTH_LEN = 2.5
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
        
        # Система трекинга рук
        self.hand_tracks = {}  # {track_id: {'hand_type': 'left'/'right'/'twohands', 'confidence_history': [], 'frames_count': 0, 'frames_since_last_seen': 0}}
        self.next_track_id = 0
        self.min_frames_for_stable_classification = 5  # Минимум кадров для стабильной классификации
        self.max_confidence_history = 10  # Максимум кадров истории для усреднения
        self.max_tracks_to_display = 2  # Максимум треков для отображения
        self.max_inactive_frames = 15  # Максимум кадров неактивности перед удалением трека
        
        # Параметры обработки
        self.CROP_SIZE = 256
        self.map_colors = {0: (255,0,0), 1: (255,128,0), 2: (102,255,255), 3: (153,0,153), 4: (178,102,55)}
        
        # YOLO классификация рук
        # Попробуем разные маппинги в зависимости от того, что возвращает YOLO
        self.yolo_classes = {0: 'hand_l', 1: 'hand_r', 2: 'twohands'}  # Альтернативный маппинг
        
        # ToF параметры
        DEPTH_LEN = 2.5
        self.TOF_PARAMS = {
            "ToF::StreamFps": 100,
            "ToF::Distance": DEPTH_LEN,
            "ToF::Exposure": 0.15,
            "ToF::DepthMedianBlur": 0,
            "ToF::DepthFlyingPixelRemoval": 2,
            "ToF::Threshold": 40,
            "ToF::Gain": 9,
            "ToF::AutoExposure": 0,
            "ToF::DepthSmoothStrength": 0,
            "ToF::DepthCompletion": 0,
        }
        
        # Параметры depth
        self.DEPTH_SCALE_ALPHA = 2000.0 / 60000.0
        self.RAW_DEPTH_MULTIPLIER = 2
        self.MAX_DEPTH_VALUE = 65535.0
        self.DEFAULT_TOF_RANGE_M = DEPTH_LEN
        
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
    
    def calculate_iou(self, box1, box2):
        """Вычисление IoU (Intersection over Union) между двумя боксами"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Вычисляем площадь пересечения
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Вычисляем площади боксов
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Вычисляем union
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def assign_track_id(self, detection_box, yolo_class):
        """Назначение track_id для новой детекции с использованием IoU"""
        detection_box = [float(x) for x in detection_box]
        
        best_track_id = None
        best_iou = 0.0
        iou_threshold = 0.3  # Минимальный IoU для сопоставления с существующим треком
        
        # Ищем трек с наибольшим IoU
        for track_id, track_info in self.hand_tracks.items():
            if track_info.get('last_box') is not None and track_info['frames_since_last_seen'] < 5:
                iou = self.calculate_iou(detection_box, track_info['last_box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
        
        # Если нашли подходящий трек, возвращаем его ID
        if best_track_id is not None and best_iou > iou_threshold:
            return best_track_id
        
        # Создаем новый трек
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def update_hand_track(self, track_id, yolo_class, confidence, detection_box):
        """Обновление информации о треке руки"""
        if track_id not in self.hand_tracks:
            self.hand_tracks[track_id] = {
                'hand_type': None,
                'confidence_history': [],
                'frames_count': 0,
                'frames_since_last_seen': 0,  # Счетчик кадров с последнего появления
                'stable_classification': None,
                'last_box': None  # Последний бокс для трекинга
            }
        
        track_info = self.hand_tracks[track_id]
        
        # Сохраняем последний бокс
        track_info['last_box'] = detection_box
        
        # Добавляем текущую классификацию в историю
        track_info['confidence_history'].append((yolo_class, confidence))
        track_info['frames_count'] += 1
        track_info['frames_since_last_seen'] = 0  # Сбрасываем счетчик неактивности
        
        # Ограничиваем историю
        if len(track_info['confidence_history']) > self.max_confidence_history:
            track_info['confidence_history'].pop(0)
        
        # Определяем стабильную классификацию
        if len(track_info['confidence_history']) >= self.min_frames_for_stable_classification:
            # Подсчитываем наиболее частый класс за последние кадры
            recent_classes = [cls for cls, conf in track_info['confidence_history'][-self.min_frames_for_stable_classification:]]
            most_common_class = max(set(recent_classes), key=recent_classes.count)
            
            # Проверяем стабильность (минимум 70% кадров должны быть одного класса)
            stability_ratio = recent_classes.count(most_common_class) / len(recent_classes)
            if stability_ratio >= 0.7:
                track_info['stable_classification'] = most_common_class
                track_info['hand_type'] = self.yolo_classes.get(most_common_class, f'unknown_{most_common_class}')
    
    def resolve_hand_conflicts(self, detections):
        """Разрешение конфликтов: не более одной левой и одной правой руки в кадре"""
        if len(detections) <= 2:
            return detections
        
        # Проверяем детекции по типам
        left_hands = [d for d in detections if d['yolo_class'] == 0]  # hand_l
        right_hands = [d for d in detections if d['yolo_class'] == 1]  # hand_r
        two_hands = [d for d in detections if d['yolo_class'] == 2]   # twohands
        
        # Конфликт: несколько левых рук
        if len(left_hands) > 1:
            print(f"DEBUG: Обнаружено {len(left_hands)} левых рук, оставляем одну с наибольшей уверенностью")
            # Оставляем только одну левую руку с наибольшей уверенностью
            best_left = max(left_hands, key=lambda d: d['confidence'])
            detections = [d for d in detections if d['yolo_class'] != 0 or d == best_left]
        
        # Конфликт: несколько правых рук
        if len(right_hands) > 1:
            print(f"DEBUG: Обнаружено {len(right_hands)} правых рук, оставляем одну с наибольшей уверенностью")
            # Оставляем только одну правую руку с наибольшей уверенностью
            best_right = max(right_hands, key=lambda d: d['confidence'])
            detections = [d for d in detections if d['yolo_class'] != 1 or d == best_right]
        
        # Конфликт: несколько "двух рук"
        if len(two_hands) > 1:
            print(f"DEBUG: Обнаружено {len(two_hands)} детекций 'двух рук', оставляем одну с наибольшей уверенностью")
            # Оставляем только одну детекцию "двух рук" с наибольшей уверенностью
            best_two_hands = max(two_hands, key=lambda d: d['confidence'])
            detections = [d for d in detections if d['yolo_class'] != 2 or d == best_two_hands]
        
        return detections
    
    def cleanup_inactive_tracks(self):
        """Очистка неактивных треков"""
        # Увеличиваем счетчик неактивности для всех треков
        for track_id, track_info in self.hand_tracks.items():
            track_info['frames_since_last_seen'] += 1
        
        # Удаляем треки, которые не появлялись слишком долго
        inactive_tracks = []
        for track_id, track_info in self.hand_tracks.items():
            if track_info['frames_since_last_seen'] > self.max_inactive_frames:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.hand_tracks[track_id]
            if track_id in self.previous_detections:
                del self.previous_detections[track_id]
    
    def filter_to_best_tracks(self, detections):
        """Фильтрация детекций: оставляем только самые долгие треки"""
        if len(detections) <= self.max_tracks_to_display:
            return detections
        
        # Сортируем треки по длительности (frames_count) в убывающем порядке
        sorted_tracks = sorted(self.hand_tracks.items(), key=lambda x: x[1]['frames_count'], reverse=True)
        
        # Берем top-N самых долгих треков
        best_track_ids = {track_id for track_id, _ in sorted_tracks[:self.max_tracks_to_display]}
        
        print(f"DEBUG: Всего треков: {len(detections)}, Оставляем только 2 самых долгих: {best_track_ids}")
        
        # Фильтруем детекции
        filtered_detections = [d for d in detections if d['track_id'] in best_track_ids]
        
        return filtered_detections
    
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
    
    def interpolate_hand_depth(self, depth, keypoints, reference_depth):
        """
        Интерполяция depth в области руки
        
        Args:
            depth: depth карта
            keypoints: список координат keypoints [(x, y), ...]
            reference_depth: опорное значение depth (от точки p_id == 0)
        
        Returns:
            tuple: (depth карта с интерполированными значениями, маска области интерполяции, convex hull)
        """
        if depth is None or len(keypoints) == 0:
            return depth, None, None
        
        depth_copy = depth.copy()
        h, w = depth.shape[:2]
        
        # Создаем маску области руки из keypoints
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Конвертируем keypoints в формат для cv2.fillPoly
        points = np.array(keypoints, dtype=np.int32)
        
        hull = None
        # Создаем выпуклую оболочку (convex hull) вокруг всех keypoints
        if len(points) >= 3:
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
            
            # Немного расширяем маску для захвата всей области
            kernel = np.ones((20, 20), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Находим области в маске где depth == 0 или слишком большой (пол)
            # Используем reference_depth как ориентир
            if reference_depth > 0:
                # Считаем что значения отличающиеся от reference_depth больше чем на 0.2м это пол
                depth_meters = depth * (DEPTH_LEN/65536)
                
                # Создаем маску плохих точек (слишком далеко от reference_depth или == 0)
                bad_depth_mask = np.zeros_like(mask)
                bad_depth_mask[(depth_meters == 0) | (np.abs(depth_meters - reference_depth) > 0.2)] = 255
                
                # Применяем маску руки - интересуют только плохие точки внутри руки
                bad_depth_mask = cv2.bitwise_and(bad_depth_mask, mask)
                
                # Создаем маску хороших точек в области руки
                good_depth_mask = cv2.bitwise_and(mask, cv2.bitwise_not(bad_depth_mask))
                
                # Если есть плохие точки, интерполируем
                if np.any(bad_depth_mask > 0) and np.any(good_depth_mask > 0):
                    # Используем inpaint для заполнения плохих областей
                    depth_float = depth.astype(np.float32)
                    depth_inpainted = cv2.inpaint(depth_float, bad_depth_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    
                    # Применяем результат только в области руки
                    depth_copy = np.where(mask > 0, depth_inpainted, depth_copy)
                    
                    # Сглаживаем результат в области руки
                    depth_copy_smoothed = cv2.GaussianBlur(depth_copy, (5, 5), 0)
                    depth_copy = np.where(mask > 0, depth_copy_smoothed, depth_copy)
        
        return depth_copy.astype(depth.dtype), mask, hull
    
    def process_frame(self, image, depth=None):
        """Обработка одного кадра"""
        # Обработка изображения как в latest.py
        if len(image.shape) == 2:  # Если изображение в оттенках серого
            ir_image_8bit = cv2.convertScaleAbs(image, alpha=self.DEPTH_SCALE_ALPHA)
            _, ir_image_8bit = cv2.threshold(ir_image_8bit, 10, 60, cv2.THRESH_TOZERO)
            image = np.copy(ir_image_8bit)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:  # Если изображение уже в цвете
            image = cv2.convertScaleAbs(image, alpha=self.DEPTH_SCALE_ALPHA)
            _, image = cv2.threshold(image, 10, 60, cv2.THRESH_TOZERO)
        
        draw_image = image.copy()
        image_h, image_w = image.shape[:2]
        
        # Инференс YOLO
        results = self.model.predict(image)
        
        # DEBUG: Проверяем, что возвращает YOLO модель
        if len(results[0].boxes) > 0:
            print(f"DEBUG: YOLO detected {len(results[0].boxes)} objects")
            for i, cls in enumerate(results[0].boxes.cls):
                print(f"DEBUG: Object {i}: class {int(cls.item())}")
        else:
            print("DEBUG: No objects detected by YOLO")
        
        list_src_crops, resized_crops = [], []
        detections = []
        
        # Собираем все детекции с информацией о треках
        for i, box in enumerate(results[0].boxes.xyxy[:2]):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(results[0].boxes.cls[i].item())
            confidence = float(results[0].boxes.conf[i].item())
            if confidence < 0.4:
                continue
            
            # Назначаем track_id
            box_coords = box.tolist()
            track_id = self.assign_track_id(box_coords, cls)
            
            # Обновляем информацию о треке (передаем координаты бокса)
            self.update_hand_track(track_id, cls, confidence, box_coords)
            
            # Добавляем детекцию в список
            detections.append({
                'box': [x1, y1, x2, y2],
                'yolo_class': cls,
                'confidence': confidence,
                'track_id': track_id,
                'index': i
            })
        
        # Разрешаем конфликты между левой и правой рукой
        detections = self.resolve_hand_conflicts(detections)
        
        # Очищаем неактивные треки
        self.cleanup_inactive_tracks()
        
        # Оставляем только самые долгие треки (защита от ложных детекций)
        detections = self.filter_to_best_tracks(detections)
        
        # Обрабатываем финальные детекции
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            cls = detection['yolo_class']
            track_id = detection['track_id']
            i = detection['index']
            
            # Получаем стабильную классификацию или используем текущую
            track_info = self.hand_tracks.get(track_id, {})
            stable_class = track_info.get('stable_classification', cls)
            
            # DEBUG: Выводим информацию о классе
            print(f"DEBUG: Track {track_id} - YOLO class: {cls}, Stable class: {stable_class}")
            print(f"DEBUG: Track frames count: {track_info.get('frames_count', 0)}")
            
            # Используем стабильную классификацию если она есть
            final_class = stable_class if stable_class is not None else cls
            yolo_class_name = self.yolo_classes.get(final_class, f'unknown_{final_class}')
            
            print(f"DEBUG: Final class: {final_class}, Mapped to: {yolo_class_name}")
            
            cropped_hand = image[y1:y2, x1:x2]
            list_src_crops.append(cropped_hand)
            cropped_resized = cv2.resize(cropped_hand, (self.CROP_SIZE, self.CROP_SIZE))
            resized_crops.append(cropped_resized)
            
            # Расширяем бокс на 1%
            x1_smooth, y1_smooth, x2_smooth, y2_smooth = x1, y1, x2, y2
            expand_width = int(0.01 * (x2_smooth - x1_smooth))
            expand_height = int(0.01 * (y2_smooth - y1_smooth))
            
            x1_smooth = max(0, x1_smooth - expand_width)
            y1_smooth = max(0, y1_smooth - expand_height)
            x2_smooth += expand_width
            y2_smooth += expand_height
            
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
                print(j, pt[0], pt[1], pt[2])
                self.sender.send(address=f"/bboxes/bbox_{i}/point_{j}", data=[pt[0], pt[1], pt[2]])
            
            # Отправка информации о классе детекции
            self.sender.send(address=f"/hand_{yolo_class_name}/class", data=[final_class, yolo_class_name])
        
        # Если нет детекций, возвращаем оригинальное изображение
        if len(list_src_crops) == 0:
            return draw_image, [], None
        
        # Предсказание keypoints и классификация
        with torch.no_grad():
            output = run_model_batch(resized_crops, self.kps_model, image_size=256)
        kps_preds = output[0]
        
        cls_output = run_model_batch(resized_crops, self.trt_model, image_size=256)[0]
        
        # Первый проход: собираем все keypoints для каждой руки и получаем reference depth
        hands_data = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['box']
            
            preds = np.expand_dims(kps_preds[i], 0)[:, :, :2] * 256
            preds = preds[0]
            h, w, _ = list_src_crops[i].shape
            
            # Собираем все абсолютные координаты keypoints
            abs_keypoints = []
            reference_depth = 0
            
            for p_id, (x_norm, y_norm) in enumerate(preds):
                x_norm, y_norm = x_norm / self.CROP_SIZE, y_norm / self.CROP_SIZE
                px = int(x_norm * w)
                py = int(y_norm * h)
                abs_px = x1 + px
                abs_py = y1 + py
                abs_keypoints.append((abs_px, abs_py))
                
                # Получаем reference depth от точки p_id == 0 (основание ладони)
                if p_id == 0 and depth is not None:
                    if 0 <= abs_py < depth.shape[0] and 0 <= abs_px < depth.shape[1]:
                        reference_depth = depth[abs_py, abs_px] * (DEPTH_LEN/65536)
            
            hands_data.append({
                'detection': detection,
                'keypoints': abs_keypoints,
                'reference_depth': reference_depth,
                'preds': preds,
                'crop_h': h,
                'crop_w': w
            })
        
        # Интерполируем depth для каждой руки
        interpolated_depth = depth.copy() if depth is not None else None
        interpolation_masks = []  # Маски областей интерполяции для визуализации
        interpolation_hulls = []  # Convex hulls для визуализации
        
        if interpolated_depth is not None:
            for hand_data in hands_data:
                if hand_data['reference_depth'] > 0:
                    interpolated_depth, mask, hull = self.interpolate_hand_depth(
                        interpolated_depth,
                        hand_data['keypoints'],
                        hand_data['reference_depth']
                    )
                    if mask is not None:
                        interpolation_masks.append(mask)
                    if hull is not None:
                        interpolation_hulls.append(hull)
        
        # Второй проход: отрисовка результатов с интерполированным depth
        for i, hand_data in enumerate(hands_data):
            detection = hand_data['detection']
            x1, y1, x2, y2 = detection['box']
            track_id = detection['track_id']
            
            # Получаем стабильную классификацию или используем текущую
            track_info = self.hand_tracks.get(track_id, {})
            stable_class = track_info.get('stable_classification', detection['yolo_class'])
            final_class = stable_class if stable_class is not None else detection['yolo_class']
            yolo_class_name = self.yolo_classes.get(final_class, f'unknown_{final_class}')
            
            # DEBUG: Выводим информацию о классе во второй части
            print(f"DEBUG (second part): Track {track_id} - Final class: {final_class}, mapped to: {yolo_class_name}")
            
            label = cls_output[i].argmax()
            color = self.map_colors[label]
            
            preds = hand_data['preds']
            h, w = hand_data['crop_h'], hand_data['crop_w']
            points = []
            
            for p_id, (x_norm, y_norm) in enumerate(preds):
                x_norm, y_norm = x_norm / self.CROP_SIZE, y_norm / self.CROP_SIZE
                px = int(x_norm * w)
                py = int(y_norm * h)
                abs_px = x1 + px
                abs_py = y1 + py
                
                if p_id == 0:
                    my_x, my_y = abs_px, abs_py
                    z = 0
                    if interpolated_depth is not None and 0 <= my_y < interpolated_depth.shape[0] and 0 <= my_x < interpolated_depth.shape[1]:
                        z = interpolated_depth[my_y, my_x] * (DEPTH_LEN/65536)
                    cv2.circle(draw_image, (my_x, my_y), 1, (0, 255, 0), -1)
                    my_x_normal = 640 - 1 - my_x
                    my_y_normal = 480 - 1 - my_y
                    real_x, real_y, real_z = self.coord_transformer.pixel_to_floor_3d(my_x_normal, my_y_normal, z)
                    cv2.putText(draw_image, f'{real_x:.3f} {real_y:.3f}, {real_z:.3f}, {z:.3f}', (my_x, my_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                points.append((abs_px, abs_py))
                if p_id == 8 or p_id == 4:
                    if p_id == 8:
                        p8 = px, py
                    if p_id == 4:
                        p4 = px, py
                    color = (117, 0, 178)
                cv2.circle(draw_image, (abs_px, abs_py), 1, (0, 255, 0), -1)
            
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
                start = points[start_idx]
                end = points[end_idx]
                cv2.line(draw_image, start, end, (0, 255, 0), thickness=1)
            
            # Отправка каждой keypoint отдельно с 3D координатами
            for p_id, (x_norm, y_norm) in enumerate(preds):
                x_norm, y_norm = x_norm / self.CROP_SIZE, y_norm / self.CROP_SIZE
                px = int(x_norm * w)
                py = int(y_norm * h)
                abs_px = x1 + px
                abs_py = y1 + py
                
                # Получаем глубину для keypoint (используем интерполированный depth)
                z = 0
                if interpolated_depth is not None and 0 <= abs_py < interpolated_depth.shape[0] and 0 <= abs_px < interpolated_depth.shape[1]:
                    z = interpolated_depth[abs_py, abs_px] * (DEPTH_LEN/65536)
                
                # Нормализуем координаты
                my_x_normal = 640 - 1 - abs_px
                my_y_normal = 480 - 1 - abs_py
                real_x, real_y, real_z = self.coord_transformer.pixel_to_floor_3d(my_x_normal, my_y_normal, z)
                if p_id == 0:
                    cv2.putText(draw_image, f'{real_z:.3f}', (x2, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if p_id == 20:
                    cv2.putText(draw_image, f'{real_z:.3f}', (x2, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Отправка каждой keypoint отдельно с классом руки
                self.sender.send(address=f"/hand_{yolo_class_name}/keypoint_{p_id}", data=[real_x, real_y, real_z])
            
            # Отправка информации о жесте
            self.sender.send(address=f"/hand_{yolo_class_name}/gesture", data=[int(label), float(cls_output[i].max())])
            
            # Рисуем линию и bounding box
            cv2.line(draw_image, (0, 550), (550, 550), color, 2)
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_image, f'id:{track_id}, {detection['yolo_class']}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Визуализация областей интерполяции
        if len(interpolation_hulls) > 0:
            # Создаем полупрозрачный оверлей для областей интерполяции
            overlay = draw_image.copy()
            
            for i, hull in enumerate(interpolation_hulls):
                # Рисуем заполненный hull полупрозрачным цветом
                cv2.fillPoly(overlay, [hull], (0, 255, 255))  # Желтый цвет для области
                
                # Рисуем контур hull
                cv2.polylines(draw_image, [hull], True, (0, 255, 255), 2)
            
            # Смешиваем оверлей с основным изображением (30% прозрачность)
            alpha = 0.3
            draw_image = cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0)
            
            # Добавляем текст с пояснением
            cv2.putText(draw_image, 'Yellow: Interpolation area', (10, image_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return draw_image, list_src_crops, interpolated_depth
    
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
            
            # Инициализация FPS
            previousTime_FPS = 0
            currentTime_FPS = 0
            startTime = time.time()
            jjj = 0
            
            # Основной цикл обработки
            while True:
                # Получение кадров из ToF потока
                images = tof_stream.getPyImage()
                
                # Проверка валидности кадров
                if not images or len(images) < 2 or images[1] is None or images[1].size == 0:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Обработка кадров (как в latest.py)
                image = images[1]
                depth = images[0]
                cv2.imwrite('temp.png', depth)
                
                jjj += 1
                
                # Пропускаем кадры (как в latest.py)
                if jjj % 8 != 0:
                    continue
                
                # Обработка кадра
                processed_image, crops, interpolated_depth = self.process_frame(image, depth)
                
                # Отображение результата
                cv2.imshow("Hand Detection", processed_image)
                
                # Показываем оригинальное изображение
                if len(image.shape) == 2:
                    cv2.imshow("Original IR", image)
                else:
                    cv2.imshow("Original IR", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                
                # Нормализация depth для визуализации
                if depth is not None:
                    # Оригинальный depth
                    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Original", depth_colored)
                    
                    # Интерполированный depth
                    if interpolated_depth is not None:
                        depth_interpolated_normalized = cv2.normalize(interpolated_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_interpolated_colored = cv2.applyColorMap(depth_interpolated_normalized, cv2.COLORMAP_JET)
                        cv2.imshow("Depth Interpolated", depth_interpolated_colored)
                
                cv2.imwrite(f'roma_images/{jjj}.png', processed_image)
                
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
    parser.add_argument('--yolo', type=str, default="yolo_new_lr.engine",
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
