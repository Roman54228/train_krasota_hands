import os
from pathlib import Path
from PIL import Image  # Лучше для проверки изображений, чем OpenCV
import shutil

# Настройки
ROOT_DIR = Path("/media/4TB/HAGRID/hagridv2_512/crops/train_multitask")  # <-- Убедись, что путь правильный
IMAGES_DIR = ROOT_DIR / "images"
LABELS_DIR = ROOT_DIR / "labels"

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Логи
missing_label = []
missing_image = []
corrupted_image = []
deleted_files = []

def is_valid_image(image_path):
    """Проверяет, можно ли открыть изображение."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Проверяет целостность, но не загружает
        return True
    except Exception:
        return False

def clean_dataset():
    print("🔍 Начинаем проверку парности и целостности файлов...")
    all_image_files = {}
    all_label_files = {}

    # Собираем все изображения и их пути
    for img_path in IMAGES_DIR.rglob("*"):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            # Ключ: относительный путь без расширения
            rel_key = img_path.relative_to(IMAGES_DIR).with_suffix('')
            all_image_files[rel_key] = img_path

    # Собираем все .txt файлы
    for txt_path in LABELS_DIR.rglob("*.txt"):
        rel_key = txt_path.relative_to(LABELS_DIR).with_suffix('')
        all_label_files[rel_key] = txt_path

    print(f"✅ Найдено изображений: {len(all_image_files)}")
    print(f"✅ Найдено разметок: {len(all_label_files)}")

    # 1. Удалить .txt, для которых нет изображения
    for rel_key, txt_path in all_label_files.items():
        if rel_key not in all_image_files:
            print(f"❌ Нет изображения для: {txt_path}")
            missing_image.append(txt_path)
            try:
                txt_path.unlink()
                deleted_files.append(txt_path)
                print(f"🗑️ Удалён: {txt_path}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить {txt_path}: {e}")

    # 2. Удалить изображения, для которых нет .txt, и проверить на битость
    for rel_key, img_path in all_image_files.items():
        if rel_key not in all_label_files:
            print(f"❌ Нет разметки для: {img_path}")
            missing_label.append(img_path)
            try:
                img_path.unlink()
                deleted_files.append(img_path)
                print(f"🗑️ Удалён: {img_path}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить {img_path}: {e}")
        else:
            # Проверяем, читается ли изображение
            if not is_valid_image(img_path):
                print(f"💥 Изображение повреждено: {img_path}")
                corrupted_image.append(img_path)
                # Удаляем и изображение, и соответствующий .txt
                txt_path = all_label_files[rel_key]
                try:
                    img_path.unlink()
                    deleted_files.append(img_path)
                    print(f"🗑️ Удалено повреждённое изображение: {img_path}")
                except Exception as e:
                    print(f"⚠️ Не удалось удалить {img_path}: {e}")
                try:
                    txt_path.unlink()
                    deleted_files.append(txt_path)
                    print(f"🗑️ Удалена разметка (без пары): {txt_path}")
                except Exception as e:
                    print(f"⚠️ Не удалось удалить {txt_path}: {e}")

    # Статистика
    print("\n" + "="*50)
    print("✅ Проверка завершена.")
    print(f"Удалено файлов: {len(deleted_files)}")
    if missing_image:
        print(f"Отсутствующие изображения (удалены .txt): {len(missing_image)}")
    if missing_label:
        print(f"Отсутствующие разметки (удалены .png/.jpg): {len(missing_label)}")
    if corrupted_image:
        print(f"Повреждённые изображения: {len(corrupted_image)}")

    print(f"\n🗑️ Удалённые файлы:")
    for f in deleted_files:
        print(f"  {f}")

if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        print(f"❌ Папка images не найдена: {IMAGES_DIR}")
    elif not LABELS_DIR.exists():
        print(f"❌ Папка labels не найдена: {LABELS_DIR}")
    else:
        clean_dataset()