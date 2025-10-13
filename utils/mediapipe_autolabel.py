"""
Автоматическая разметка изображений рук с помощью YOLO и MediaPipe HandLandmarker.
Скрипт детектирует руки с помощью YOLO, затем извлекает ключевые точки через MediaPipe.
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import argparse
from glob import glob
from ultralytics import YOLO
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

# Настройка MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_models(yolo_path: str, mediapipe_path: str, num_hands: int = 2,
                confidence: float = 0.3) -> Tuple[YOLO, HandLandmarker]:
    """
    Загружает YOLO и MediaPipe модели.
    
    Args:
        yolo_path: Путь к модели YOLO
        mediapipe_path: Путь к модели MediaPipe HandLandmarker
        num_hands: Максимальное количество рук для детекции
        confidence: Минимальный уровень уверенности для детекции
        
    Returns:
        Кортеж (yolo_model, hand_landmarker)
    """
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Модель YOLO не найдена: {yolo_path}")
    
    if not os.path.exists(mediapipe_path):
        raise FileNotFoundError(f"Модель MediaPipe не найдена: {mediapipe_path}")
    
    print(f"Загрузка YOLO модели: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    
    print(f"Загрузка MediaPipe модели: {mediapipe_path}")
    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mediapipe_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )
    )
    
    return yolo_model, landmarker


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Препроцессинг IR-изображения: 16-бит -> 8-бит, THRESH_TOZERO, COLORMAP_PINK, flip.
    
    Args:
        image: Входное изображение (может быть 8 или 16 бит)
        
    Returns:
        Обработанное 8-битное цветное изображение
    """
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=(1000.0 / 40000.0))
    
    _, image = cv2.threshold(image, 10, 60, cv2.THRESH_TOZERO)
    image = cv2.applyColorMap(image, cv2.COLORMAP_PINK)
    image = cv2.flip(image, -1)  # Отражение по обеим осям
    return image


def draw_hand_landmarks_cv2(image: np.ndarray, hand_landmarks: List, handedness: str) -> None:
    """
    Рисует ключевые точки и соединения руки с помощью OpenCV.
    
    Args:
        image: Изображение для рисования
        hand_landmarks: Список ключевых точек руки (21 точка)
        handedness: "Left" или "Right"
    """
    h, w = image.shape[:2]
    color = (0, 255, 0) if handedness == "Left" else (255, 0, 0)

    # Соединения между ключевыми точками руки (MediaPipe hand topology)
    connections = [
        # Большой палец
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Указательный
        (5, 6), (6, 7), (7, 8),
        # Средний
        (9, 10), (10, 11), (11, 12),
        # Безымянный
        (13, 14), (14, 15), (15, 16),
        # Мизинец
        (17, 18), (18, 19), (19, 20),
        # Ладонь
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]

    # Конвертируем нормализованные координаты в пиксели
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Рисуем линии соединений
    for start_idx, end_idx in connections:
        start = points[start_idx]
        end = points[end_idx]
        cv2.line(image, start, end, color, thickness=2)

    # Рисуем точки
    for pt in points:
        cv2.circle(image, pt, radius=3, color=color, thickness=-1)

    # Добавляем текст с типом руки
    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    cv2.putText(image, handedness,
                (max(0, x_min - 100), max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)


def detect_hands_on_image(image_path: str, landmarker: HandLandmarker, yolo_model: YOLO, 
                          output_path: Optional[str] = None, expand_factor: float = 1.4) -> np.ndarray:
    """
    Обрабатывает одно изображение: детекция рук через YOLO + извлечение ключевых точек через MediaPipe.
    
    Args:
        image_path: Путь к входному изображению
        landmarker: Загруженная модель MediaPipe HandLandmarker
        yolo_model: Загруженная модель YOLO для детекции рук
        output_path: Путь для сохранения результата (опционально)
        expand_factor: Коэффициент расширения bounding box (по умолчанию 1.4 = +40%)
        
    Returns:
        Аннотированное изображение с отрисованными ключевыми точками
    """
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # Препроцессинг
    processed_image = preprocess_image(image)
    annotated_image = processed_image.copy()
    h, w = processed_image.shape[:2]

    # Детекция рук через YOLO
    results = yolo_model(processed_image, verbose=False)

    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        # Расширяем bounding box для лучшей детекции ключевых точек
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        box_w = x2 - x1
        box_h = y2 - y1

        new_w = int(box_w * expand_factor)
        new_h = int(box_h * expand_factor)

        # Новые координаты с центром в исходном боксе
        x1_new = max(0, cx - new_w // 2)
        y1_new = max(0, cy - new_h // 2)
        x2_new = min(w, cx + new_w // 2)
        y2_new = min(h, cy + new_h // 2)

        # Вырезаем расширенный кроп
        hand_crop = processed_image[y1_new:y2_new, x1_new:x2_new]
        if hand_crop.size == 0:
            continue

        # Подготавливаем для MediaPipe
        rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        # Запускаем детекцию ключевых точек
        detection_result = landmarker.detect(mp_image)
        if not detection_result.hand_landmarks:
            continue

        # Отрисовка результатов
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0].category_name

            # Преобразуем локальные координаты кропа в глобальные координаты изображения
            scaled_landmarks = []
            crop_h, crop_w = hand_crop.shape[:2]
            
            for lm in hand_landmarks:
                x_local = int(lm.x * crop_w)
                y_local = int(lm.y * crop_h)
                x_global = x1_new + x_local
                y_global = y1_new + y_local
                
                # Нормализуем координаты относительно полного изображения
                scaled_landmarks.append(
                    type('obj', (object,), {
                        'x': x_global / w,
                        'y': y_global / h,
                        'z': lm.z
                    })
                )

            draw_hand_landmarks_cv2(annotated_image, scaled_landmarks, handedness)
            
            # Рисуем расширенный bounding box
            cv2.rectangle(annotated_image, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 2)
    
    # Сохранение результата
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
    
    return annotated_image


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Автоматическая разметка изображений рук с помощью YOLO и MediaPipe"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="my_experiments/run18/weights/best.pt",
        help="Путь к модели YOLO для детекции рук"
    )
    parser.add_argument(
        "--mediapipe-model",
        type=str,
        default="./hand_landmarker.task",
        help="Путь к модели MediaPipe HandLandmarker"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Папка с входными изображениями"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="output_annotations",
        help="Папка для сохранения результатов"
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Максимальное количество рук для детекции"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Минимальный уровень уверенности для детекции"
    )
    parser.add_argument(
        "--expand-factor",
        type=float,
        default=1.4,
        help="Коэффициент расширения bounding box (1.4 = +40%%)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"],
        help="Расширения файлов для обработки"
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Очистить выходную папку перед началом"
    )
    
    return parser.parse_args()


def main():
    """Основная функция для обработки изображений."""
    args = parse_args()
    
    # Проверка входной папки
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Входная папка не найдена: {args.input_folder}")
    
    # Подготовка выходной папки
    if args.clear_output and os.path.exists(args.output_folder):
        print(f"Очистка выходной папки: {args.output_folder}")
        shutil.rmtree(args.output_folder, ignore_errors=True)
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Загрузка моделей
    print("\n=== Загрузка моделей ===")
    yolo_model, landmarker = load_models(
        args.yolo_model,
        args.mediapipe_model,
        args.num_hands,
        args.confidence
    )
    
    # Сбор изображений
    print(f"\n=== Поиск изображений в {args.input_folder} ===")
    image_files = []
    for ext in args.extensions:
        image_files.extend(glob(os.path.join(args.input_folder, ext)))
    
    if not image_files:
        print(f"Не найдено изображений с расширениями: {args.extensions}")
        return
    
    # Сортировка (пробуем сортировать по числовым индексам, если не получается - по имени)
    try:
        image_paths = sorted(
            image_files,
            key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1].replace('frame', ''))
        )
    except (ValueError, IndexError):
        image_paths = sorted(image_files)
    
    print(f"Найдено изображений: {len(image_paths)}")
    
    # Обработка изображений
    print("\n=== Обработка изображений ===")
    for i, img_path in enumerate(image_paths):
        try:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            out_path = os.path.join(args.output_folder, f"{i:05d}_{name}.png")
            
            detect_hands_on_image(
                img_path,
                landmarker,
                yolo_model,
                out_path,
                args.expand_factor
            )
            
            print(f"[{i+1}/{len(image_paths)}] Обработано: {filename} -> {os.path.basename(out_path)}")
            
        except Exception as e:
            print(f"[{i+1}/{len(image_paths)}] Ошибка при обработке {img_path}: {e}")
            continue
    
    print(f"\n=== Завершено! Результаты сохранены в: {args.output_folder} ===")


if __name__ == "__main__":
    main()