#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorRT Utils - Утилиты для работы с TensorRT моделями

Модуль для загрузки и работы с TensorRT моделями для детекции рук,
keypoints и классификации жестов.

Автор: Extracted from process_images.py and process_camera.py
Дата: 2025
"""

import os
import time
import numpy as np
import cv2
from PIL import Image
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def load_hand_gesture_model(engine_path: str) -> TrtRunner:
    """
    Загрузка TensorRT модели
    
    Args:
        engine_path: Путь к .engine файлу
        
    Returns:
        TrtRunner: Загруженная модель
        
    Raises:
        FileNotFoundError: Если файл модели не найден
    """
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    with open(engine_path, "rb") as f:
        engine = EngineFromBytes(f.read())
    runner = TrtRunner(engine)
    print(f"Model loaded from {engine_path}")
    return runner


def run_model_batch(images, runner: TrtRunner, image_size=256):
    """
    Принимает список изображений (1, 2, ... N) и делает inference батчом.

    Args:
        images: Список изображений (PIL.Image или numpy.ndarray)
        runner: TrtRunner модель
        image_size: Размер изображения для обработки

    Returns:
        tuple: (outputs, inference_time)
            - outputs: Результаты модели [N, 21, 3] для keypoints или [N, num_classes] для классификации
            - inference_time: Время выполнения в секундах
    """
    batch = []
    for img in images:
        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image.Image):
            img = np.array(img)
        else:
            raise ValueError("Images must be PIL.Image or numpy.ndarray")

        img = cv2.resize(img, (image_size, image_size))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, H, W]
        batch.append(img)

    input_tensor = np.stack(batch)  # [N, 3, 256, 256]

    with runner:
        input_name = runner.engine[0]
        output_name = runner.engine[1]

        start_time = time.time()
        outputs = runner.infer({input_name: input_tensor})
        inference_time = time.time() - start_time

    return outputs[output_name], inference_time  # [N, 21, 3] or [N, num_classes]


def run_model_once(image, runner: TrtRunner, image_size=256):
    """
    Делает один прогон модели TensorRT через Polygraphy.
    Совместимо с polygraphy >= 1.0.0.
    
    Args:
        image: Изображение (PIL.Image или numpy.ndarray)
        runner: TrtRunner модель
        image_size: Размер изображения для обработки
        
    Returns:
        tuple: (output, inference_time)
            - output: Результат модели
            - inference_time: Время выполнения в секундах
    """
    # Подготовка изображения
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise ValueError("Изображение должно быть PIL.Image или numpy.ndarray")

    # Убедимся, что RGB
    if img.ndim == 3 and img.shape[-1] == 3:
        pass  # предполагаем RGB
    else:
        raise ValueError("Изображение должно быть 3-канальным (RGB)")

    # Изменение размера
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)  # (256, 256, 3)

    # HWC -> CHW, нормализация, float32
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, 256, 256]

    # Добавляем batch: [1, 3, 256, 256]
    input_tensor = np.expand_dims(img, axis=0)

    # === ИСПОЛЬЗУЕМ КОНТЕКСТНЫЙ МЕНЕДЖЕР И ПОЛУЧАЕМ ИМЕНА ЧЕРЕЗ API ===
    with runner:
        # Получаем имя первого входа
        input_name = runner.engine[0]  # или runner.engine.inputs[0].name
        output_name = runner.engine[1]  # или runner.engine.outputs[0].name

        start_time = time.time()
        outputs = runner.infer({input_name: input_tensor})
        inference_time = time.time() - start_time

    # Получаем результат по имени выхода
    keypoints_3d = outputs[output_name]  # shape: [1, 21, 3]

    return keypoints_3d[0], inference_time  # возвращаем [21, 3]


def predict_hand_gesture(image: np.ndarray, model: TrtRunner, input_size=(400, 400)):
    """
    Предсказание жеста руки
    
    Args:
        image: Изображение руки
        model: TrtRunner модель для классификации
        input_size: Размер входа модели
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    if image.ndim == 3 and image.shape[2] == 3:
        if image[0, 0][0] > image[0, 0][2]:  # BGR?
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input image must be HxWx3")

    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize(input_size, Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
    img_np = np.expand_dims(img_np, axis=0)   # (1, 3, H, W)

    with model:
        outputs = model.infer(feed_dict={"input": img_np})

    output = outputs["output"].squeeze()
    confidence = float(np.max(output))
    predicted_class = int(np.argmax(output))

    return predicted_class, confidence
