from ultralytics import YOLO
import tensorrt as trt
import torch

# Загрузка модели YOLO
model = YOLO('yolov8n.pt')  # Укажите путь к вашей модели

# Экспорт в TensorRT
model.export(format='engine', half=True, workspace=4)  # workspace в ГБ

print("Конвертация завершена! Файл .engine создан.")