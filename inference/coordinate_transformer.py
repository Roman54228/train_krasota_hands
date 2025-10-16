#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinate Transformer - Модуль для преобразования координат

Модуль для преобразования координат между пикселями и мировыми координатами
с поддержкой 3D преобразований и работы с depth данными.

Автор: Extracted from process_images.py and process_camera.py
Дата: 2025
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class CoordinateTransformer:
    """Класс для преобразования координат между пикселями и мировыми координатами"""
    
    def __init__(self):
        # Параметры камеры
        self.K = np.array([[529.141724, 0, 329.699677],
                          [0, 528.957397, 229.188171],
                          [0, 0, 1]], dtype=np.float64)
        
        # Параметры дисторсии
        self.dist = np.array([-0.2333385, 0.102841, -0.017275, 0.000101, 0.000010])
        
        self.height = 0.93
        
        # Угол наклона камеры (в радианах)
        self.alpha = np.deg2rad(6)
        
        # Матрица гомографии
        self.H = self._calculate_homography()
        
        # Матрица проекции для 3D преобразований
        self.RT = self._calculate_projection_matrix()
    
    def _calculate_homography(self) -> np.ndarray:
        """Расчет матрицы гомографии"""
        # Матрица поворота камеры
        R_down = np.array([[1, 0, 0],
                          [0, np.cos(self.alpha), -np.sin(self.alpha)],
                          [0, np.sin(self.alpha), np.cos(self.alpha)]], dtype=np.float64)
        
        # Вектор трансляции
        t = np.array([[0], [0], [self.height]])
        
        # Матрица проекции
        RT = np.hstack((R_down, t))
        
        # Гомография для плоскости Z=0
        H = self.K @ RT[:, [0, 1, 3]]
        
        return H
    
    def _calculate_projection_matrix(self) -> np.ndarray:
        """Расчет матрицы проекции для 3D преобразований"""
        # Матрица поворота камеры
        R_down = np.array([[1, 0, 0],
                          [0, np.cos(self.alpha), -np.sin(self.alpha)],
                          [0, np.sin(self.alpha), np.cos(self.alpha)]], dtype=np.float64)
        
        # Вектор трансляции
        t = np.array([[0], [0], [self.height]])
        
        # Матрица проекции [R|t]
        RT = np.hstack((R_down, t))
        
        return RT
    
    def pixel_to_world_3d(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """Преобразование пиксельных координат в 3D мировые координаты
        
        Args:
            u: Координата X в пикселях
            v: Координата Y в пикселях
            z: Глубина Z (расстояние от камеры)
            
        Returns:
            Tuple[float, float, float]: Мировые координаты (X, Y, Z)
        """
        # Устранение дисторсии
        pixel = np.array([[u, v]], dtype=np.float64)
        pixel = pixel.reshape(-1, 1, 2)
        pixel_undistorted = cv2.undistortPoints(pixel, self.K, self.dist, P=self.K)
        u_undist, v_undist = pixel_undistorted[0][0]

        # Преобразование нормализованных координат в 3D координаты камеры
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_cam = (u_undist - cx) / fx
        y_cam = (v_undist - cy) / fy
        
        scale = z / np.sqrt(x_cam**2 + y_cam**2 + 1)
        x_cam = x_cam * scale
        y_cam = y_cam * scale
        z_cam = scale
                
        return x_cam, y_cam, z_cam
    
    def pixel_to_world(self, u: float, v: float, z: Optional[float] = None) -> Tuple[float, float]:
        """Преобразование пиксельных координат в мировые
        
        Args:
            u: Координата X в пикселях
            v: Координата Y в пикселях  
            z: Опциональная глубина Z (если None, используется гомография для Z=0)
            
        Returns:
            Tuple[float, float]: Мировые координаты (X, Y)
        """
        if z is not None:
            # Используем 3D преобразование с учетом глубины
            x, y, _ = self.pixel_to_world_3d(u, v, z)
            return x, y
        
        # Оригинальный метод с гомографией для Z=0
        pixel = np.array([[u, v]], dtype=np.float64)
        pixel = pixel.reshape(-1, 1, 2)
        
        # Устранение дисторсии
        pixel = cv2.undistortPoints(pixel, self.K, self.dist, P=self.K)
        
        x_norm, y_norm = pixel[0][0]
        pixel = np.array([x_norm, y_norm, 1.0])
        
        # Преобразование в мировые координаты
        world = np.linalg.inv(self.H) @ pixel
        world /= world[2]  # нормализация
        
        return world[0], world[1]
    
    def camera_to_floor_coordinates(self, x_cam: float, y_cam: float, z_cam: float) -> Tuple[float, float, float]:
        """Преобразование координат камеры в систему координат пола
        
        Args:
            x_cam: X координата в системе камеры (вправо/влево)
            y_cam: Y координата в системе камеры (вверх/вниз) 
            z_cam: Z координата в системе камеры (вперед/назад)
            
        Returns:
            Tuple[float, float, float]: Координаты пола (X, Y, Z)
            - X, Y - координаты на плоскости пола
            - Z - высота точки над полом
        """
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        
        y_rot = y_cam * cos_alpha + z_cam * sin_alpha   # вертикальное смещение вниз
        z_rot = -y_cam * sin_alpha + z_cam * cos_alpha  # горизонтальное вперёд
        
        x_floor = x_cam
        y_floor = z_rot
        z_floor = self.height - y_rot  # вычитаем, потому что y_rot — вниз
        
        return x_floor, y_floor, z_floor
    
    def pixel_to_floor_3d(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """Преобразование пиксельных координат в координаты пола
        
        Args:
            u: Координата X в пикселях
            v: Координата Y в пикселях
            z: Глубина Z (расстояние от камеры в метрах)
            
        Returns:
            Tuple[float, float, float]: Координаты пола (X, Y, Z)
            - X, Y - координаты на плоскости пола
            - Z - высота точки над полом
        """
        # Сначала получаем координаты камеры
        x_cam, y_cam, z_cam = self.pixel_to_world_3d(u, v, z)
        
        # Затем преобразуем в координаты пола
        return self.camera_to_floor_coordinates(x_cam, y_cam, z_cam)
    
    def create_bbox_points(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[Tuple[float, float, float], ...]:
        """Создание точек bounding box для отправки через OSC
        
        Args:
            x1, y1, x2, y2: Координаты bounding box
            
        Returns:
            Tuple с точками: (центр, левый_верх, правый_верх, левый_низ, правый_низ)
        """
        z = 0.0
        
        # Расширение бокса на 1%
        expand_width = int(0.01 * (x2 - x1))
        expand_height = int(0.01 * (y2 - y1))
        
        x1_expanded = max(0, x1 - expand_width)
        y1_expanded = max(0, y1 - expand_height)
        x2_expanded = x2 + expand_width
        y2_expanded = y2 + expand_height
        
        # Точки углов
        pt_tl = [float(x1_expanded), float(y1_expanded), z]
        pt_tr = [float(x2_expanded), float(y1_expanded), z]
        pt_bl = [float(x1_expanded), float(y2_expanded), z]
        pt_br = [float(x2_expanded), float(y2_expanded), z]
        
        # Центральная точка
        vctl = np.array(pt_tl)
        vcbr = np.array(pt_br)
        pt_ct = (vcbr + vctl) / 2
        
        return (pt_ct, pt_tl, pt_tr, pt_bl, pt_br)
    
    def get_world_coordinates_for_keypoint(self, keypoint_x: int, keypoint_y: int, depth: Optional[float] = None) -> Tuple[float, float]:
        """Получение мировых координат для ключевой точки
        
        Args:
            keypoint_x: X координата ключевой точки в пикселях
            keypoint_y: Y координата ключевой точки в пикселях
            depth: Опциональная глубина Z
            
        Returns:
            Tuple[float, float]: Мировые координаты (X, Y)
        """
        return self.pixel_to_world(keypoint_x, keypoint_y, depth)
    
    def get_world_coordinates_3d(self, keypoint_x: int, keypoint_y: int, depth: float) -> Tuple[float, float, float]:
        """Получение полных 3D мировых координат для ключевой точки
        
        Args:
            keypoint_x: X координата ключевой точки в пикселях
            keypoint_y: Y координата ключевой точки в пикселях
            depth: Глубина Z (расстояние от камеры)
            
        Returns:
            Tuple[float, float, float]: Мировые координаты (X, Y, Z)
        """
        return self.pixel_to_floor_3d(keypoint_x, keypoint_y, depth)
