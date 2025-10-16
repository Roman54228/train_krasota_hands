#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OSC Sender - Модуль для отправки данных через OSC

Модуль для отправки данных детекции рук и keypoints через OSC протокол
с поддержкой логирования и настройки параметров.

Автор: Extracted from process_images.py and process_camera.py
Дата: 2025
"""

import logging
import numpy as np
from pythonosc import udp_client


class Sender:
    """Класс для отправки данных через OSC протокол"""
    
    def __init__(self, ip="127.0.0.1", port=5005, logging_level="DEBUG"):
        """
        Инициализация OSC отправителя
        
        Args:
            ip: IP адрес для отправки OSC сообщений
            port: Порт для отправки OSC сообщений
            logging_level: Уровень логирования (DEBUG, INFO, WARN, ERROR, CRITICAL)
        """
        # setup OSC client
        self._client = udp_client.SimpleUDPClient(ip, port)
        
        # setup logger
        self.logger = logging.getLogger("osc_sender")
        
        match logging_level:
            case "DEBUG":
                self.logger.setLevel(logging.DEBUG)
            case "INFO":
                self.logger.setLevel(logging.INFO)
            case "WARN":
                self.logger.setLevel(logging.WARN)
            case "ERROR":
                self.logger.setLevel(logging.ERROR)
            case "CRITICAL":
                self.logger.setLevel(logging.CRITICAL)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info(f"OSC Sender with ip {ip} and port {port} is ready")

    def send_hands(self, hands_data, GMHD_hands):
        """
        Отправка данных о руках через OSC
        
        Args:
            hands_data: Данные о руках (handedness, landmarks)
            GMHD_hands: 3D данные о руках
        """
        for i, hand in enumerate(hands_data.handedness):
            for j, landmark in enumerate(hands_data.hand_landmarks[i]):
                try:
                    # TODO: Add hand ID, now we can sand only one right and one right hand. Need additional data?
                    self.send(f"/hands/{hand[0].display_name}/0/{j}", [j, landmark.x, landmark.y, GMHD_hands[i].joints[j].z])

                except Exception as e:
                    self.logger.warning(e)
            
            self.logger.debug("\n")

    def send(self, address: str = "/hadns/right/person_number/handmark_number", data: list = []):
        """
        Отправка OSC сообщения
        
        Args:
            address: OSC адрес для отправки
            data: Данные для отправки
        """
        self.logger.debug(f"{address} {data}")
        self._client.send_message(address, data)
    
    def send_bbox(self, bbox_id: int, point_id: int, x: float, y: float, z: float):
        """
        Отправка данных о bounding box
        
        Args:
            bbox_id: ID bounding box
            point_id: ID точки (0=центр, 1=левый_верх, 2=правый_верх, 3=левый_низ, 4=правый_низ)
            x, y, z: Координаты точки
        """
        self.send(f"/bboxes/bbox_{bbox_id}/point_{point_id}", [x, y, z])
    
    def send_keypoint(self, hand_id: int, keypoint_id: int, x: float, y: float, z: float):
        """
        Отправка данных о ключевой точке
        
        Args:
            hand_id: ID руки
            keypoint_id: ID ключевой точки
            x, y, z: Координаты точки
        """
        self.send(f"/hands/hand_{hand_id}/keypoint_{keypoint_id}", [x, y, z])
    
    def send_gesture(self, hand_id: int, gesture_class: int, confidence: float):
        """
        Отправка данных о жесте
        
        Args:
            hand_id: ID руки
            gesture_class: Класс жеста
            confidence: Уверенность в классификации
        """
        self.send(f"/hands/hand_{hand_id}/gesture", [gesture_class, confidence])
    
    def send_hand_detection(self, bbox_id: int, gesture_class: int, confidence: float, keypoints: np.ndarray):
        """
        Отправка полных данных о детекции руки
        
        Args:
            bbox_id: ID bounding box
            gesture_class: Класс жеста
            confidence: Уверенность в классификации
            keypoints: Массив keypoints [21, 3] (x, y, z)
        """
        # Отправка классификации
        self.send(f"/hands/bbox_{bbox_id}/gesture", [gesture_class, confidence])
        
        # Отправка всех 21 keypoints
        for kp_id in range(21):
            if kp_id < len(keypoints):
                x, y, z = keypoints[kp_id]
                self.send(f"/hands/bbox_{bbox_id}/keypoint_{kp_id}", [x, y, z])
    
    def send_bbox_classification(self, bbox_id: int, gesture_class: int, confidence: float):
        """
        Отправка только классификации для bbox
        
        Args:
            bbox_id: ID bounding box
            gesture_class: Класс жеста
            confidence: Уверенность в классификации
        """
        self.send(f"/hands/bbox_{bbox_id}/classification", [gesture_class, confidence])
    
    def send_bbox_keypoints(self, bbox_id: int, keypoints: np.ndarray):
        """
        Отправка keypoints для bbox
        
        Args:
            bbox_id: ID bounding box
            keypoints: Массив keypoints [21, 3] (x, y, z)
        """
        for kp_id in range(21):
            if kp_id < len(keypoints):
                x, y, z = keypoints[kp_id]
                self.send(f"/hands/bbox_{bbox_id}/keypoint_{kp_id}", [x, y, z])
