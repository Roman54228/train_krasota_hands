# Детекция рук и кейпоинтов

Проект для детекции рук, распознавания жестов и оценки ключевых точек кисти. Поддерживает обучение моделей, инференс с трекингом и экспорт в ONNX/TensorRT.

## Возможности

- **Детекция рук** с использованием YOLO
- **Распознавание жестов** (5 классов: 2_hands, fist, normal_hand, pinch, pointer)
- **Оценка ключевых точек** (21 точка на руку)
- **Мультитаск обучение** (классификация жестов + нахождение ключевых точек на кропе)
- **Трекинг** с использованием IOU + фильтра Калмана
- **Экспорт моделей** в ONNX и TensorRT (FP16/INT8)
- **Аугментация данных** с сохранением ключевых точек
- **Интеграция с ClearML** для отслеживания экспериментов

## Структура проекта

```
train_krasota_hands/
├── training/              # Скрипты обучения
│   ├── train_classifier.py      # Обучение классификатора жестов
│   ├── train_keypoints.py       # Обучение детектора ключевых точек
│   └── train_multitask.py       # Мультитаск обучение
│
├── inference/             # Скрипты инференса
│   └── batch_inference.py       # Батч-инференс с трекингом
│
├── utils/                 # Утилиты
│   ├── augment_dataset.py       # Аугментация данных
│   ├── update_labels.py         # Обновление меток классификатором
│   └── download_data.py         # Загрузка данных с Roboflow
│
├── export/                # Экспорт моделей
│   ├── export_to_onnx.py        # Экспорт в ONNX
│   └── export_to_tensorrt.py    # Экспорт в TensorRT
│
├── requirements.txt       # Зависимости
├── .gitignore
└── README.md
```

## Установка

### 1. Клонировать репозиторий
```bash
git clone https://github.com/yourusername/train_krasota_hands.git
cd train_krasota_hands
```

### 2. Создать виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

## Подготовка данных

### Формат данных

Проект использует формат YOLO-Pose:
```
<class_id> <x_center> <y_center> <width> <height> <x1> <y1> <v1> <x2> <y2> <v2> ... <x21> <y21> <v21>
```

Структура датасета:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### Загрузка данных с Roboflow

```bash
python utils/download_data.py \
    --api_key YOUR_API_KEY \
    --workspace workspace_name \
    --project project_name \
    --version 1 \
    --format yolov11 \
    --output_dir ./dataset
```

### Аугментация данных

```bash
python utils/augment_dataset.py \
    --images_dir dataset/train/images \
    --labels_dir dataset/train/labels \
    --output_dir augmented_dataset \
    --num_augmentations 10 \
    --aug_type standard \
    --save_original
```

Доступные типы аугментации:
- `light` - легкие трансформации (яркость, размытие)
- `standard` - стандартный набор (флипы, повороты, цвет)
- `heavy` - агрессивные аугментации (искажения, дропауты)

## Обучение

### Обучение классификатора жестов

```bash
python training/train_classifier.py \
    --data_dir dataset/train \
    --num_classes 5 \
    --model_name mobilenetv3_small_100 \
    --pretrained \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir checkpoints/classifier \
    --project_name hand_gestures \
    --task_name mobilenetv3_training
```

### Обучение детектора ключевых точек

```bash
python training/train_keypoints.py \
    --train_images dataset/train/images \
    --train_labels dataset/train/labels \
    --val_images dataset/val/images \
    --val_labels dataset/val/labels \
    --model_type blazehand \
    --pretrained_weights blazehand_landmark.pth \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-5 \
    --use_hard_mining \
    --checkpoint_dir checkpoints/keypoints
```

### Мультитаск обучение (жесты + ключевые точки)

```bash
python training/train_multitask.py \
    --train_images dataset/train/images \
    --train_labels dataset/train/labels \
    --val_images dataset/val/images \
    --val_labels dataset/val/labels \
    --model_type blazehand \
    --pretrained_weights blazehand_landmark.pth \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-5 \
    --kps_weight 1.0 \
    --cls_weight 2.0 \
    --use_weighted_sampler \
    --class_weights 2.0 2.0 1.0 2.0 8.0 \
    --checkpoint_dir checkpoints/multitask
```

## 🔮 Инференс

### Батч-инференс с трекингом

```bash
python inference/batch_inference.py \
    --input_dir path/to/images \
    --output_dir results \
    --yolo_model path/to/yolo.pt \
    --model_weights checkpoints/best_multitask.pth \
    --model_type blazehand \
    --img_size 256 \
    --max_hands 4 \
    --draw_skeleton \
    --track_max_age 3 \
    --track_iou_threshold 0.3
```

Параметры:
- `--max_hands` - максимальное количество рук на кадре
- `--expand_ratio` - коэффициент расширения bbox (по умолчанию 0.3)
- `--grayscale` - конвертация в grayscale
- `--draw_skeleton` - отрисовка скелета руки
- `--track_max_age` - максимальный возраст трека
- `--track_iou_threshold` - порог IOU для сопоставления треков

## Экспорт моделей

### Экспорт в ONNX

```bash
python export/export_to_onnx.py \
    --model_type blazehand \
    --weights checkpoints/best_multitask.pth \
    --output_path models/multitask.onnx \
    --img_size 256 \
    --batch_size 1 \
    --dynamic_batch \
    --verify
```

### Экспорт в TensorRT

**FP16:**
```bash
python export/export_to_tensorrt.py \
    --onnx_path models/multitask.onnx \
    --output_path models/multitask_fp16.engine \
    --img_size 256 \
    --fp16 \
    --min_batch 1 \
    --opt_batch 2 \
    --max_batch 4
```

**INT8 с калибровкой:**
```bash
python export/export_to_tensorrt.py \
    --onnx_path models/multitask.onnx \
    --output_path models/multitask_int8.engine \
    --img_size 256 \
    --int8 \
    --calib_images dataset/val/images \
    --calib_batch_size 1 \
    --max_calib_images 500 \
    --calib_cache calibration.cache \
    --min_batch 1 \
    --opt_batch 2 \
    --max_batch 4
```

## 🛠️ Утилиты

### Обновление меток классификатором

Используется для создания мультитаск датасета из датасета ключевых точек:

```bash
python utils/update_labels.py \
    --images_dir dataset/train/images \
    --labels_dir dataset/train/labels \
    --model_weights checkpoints/best_classifier.pth \
    --model_type timm \
    --model_name mobilenetv3_small_100 \
    --num_classes 5 \
    --img_size 224 \
    --verbose
```

## Мониторинг экспериментов

Проект интегрирован с **ClearML** для отслеживания экспериментов:

1. Настройте ClearML:
```bash
clearml-init
```

2. При обучении автоматически логируются:
   - Гиперпараметры
   - Метрики (loss, accuracy)
   - Confusion matrix
   - Графики обучения

3. Просмотр результатов в веб-интерфейсе ClearML

## 🎯 Классы жестов

| Класс | ID | Описание |
|-------|----|----|
| 2_hands | 0 | Две руки |
| fist | 1 | Кулак |
| normal_hand | 2 | Обычная рука |
| pinch | 3 | Щепотка |
| pointer | 4 | Указательный палец |


