
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import timm
import numpy as np
import cv2
import random
from pathlib import Path


import timm
import torch.nn as nn
# from MediaPipePyTorch.blazebase import BlazeLandmark, BlazeBlock
from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark

import heapq
import copy

# Класс для хранения hardest примеров
import heapq
import copy


class RepeatedHardDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, hard_samples, repeat_times=3, loss_weight=2.0):
        self.base_dataset = base_dataset
        self.hard_samples = hard_samples
        self.repeat_times = repeat_times
        self.loss_weight = loss_weight
        self.len_base = len(base_dataset)
        self.len_hard = len(hard_samples) * repeat_times

    def __len__(self):
        return self.len_base + self.len_hard

    def __getitem__(self, idx):
        if idx < self.len_base:
            img, kp, _ = self.base_dataset[idx]
            return img, kp, 1.0  # обычный вес
        else:
            hard_idx = (idx - self.len_base) % len(self.hard_samples)
            img, kp = self.hard_samples[hard_idx]
            return img.clone(), kp.clone(), self.loss_weight  # повышенный вес
        
        
class HardSampleManager:
    def __init__(self, top_percent=0.2, repeat_times=3, loss_weight=2.0):
        self.top_percent = top_percent
        self.repeat_times = repeat_times
        self.loss_weight = loss_weight
        self.hard_samples = []

    def update(self, all_losses, all_images, all_keypoints):
        """
        Обновляем список hardest сэмплов
        """
        # Сортируем по лоссу
        indexed_losses = list(enumerate(all_losses))
        indexed_losses.sort(key=lambda x: x[1], reverse=True)

        num_hard = int(len(all_losses) * self.top_percent)
        top_indices = [i for i, _ in indexed_losses[:num_hard]]

        # Сохраняем hardest образцы
        self.hard_samples = [(all_images[i], all_keypoints[i]) for i in top_indices]

    def get_dataset(self, base_dataset):
        return RepeatedHardDataset(base_dataset, self.hard_samples, self.repeat_times, self.loss_weight)
    
class HandKeypointNet(nn.Module):
    def __init__(self, num_keypoints=21):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Бэкбон
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=False, num_classes=0)
        
        # Получаем размер фичей после backbone (например, resnet18 -> 512)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            out = self.backbone(dummy_input)
            # breakpoint()
            feature_dim = 2048  # например, 512 x 8 x 8 = 32768

        # Регрессор
        self.regressor = nn.Sequential(
            nn.Flatten(),  # [B, 512, H', W'] → [B, D]
            nn.Linear(feature_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, num_keypoints * 2),  # [B, 42]
            # nn.Sigmoid()  # нормализуем в [0..1]
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, 512, 1, 1], если num_classes=0
        return self.regressor(features).view(-1, self.num_keypoints, 2)  # [B, 21, 2]



import os
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from pathlib import Path

class HandKeypointsDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_size=(400, 400), num_keypoints=21):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.num_keypoints = num_keypoints

        # Список всех файлов
        self.image_files = []
        # breakpoint()
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = os.path.join(
            self.label_dir,
            Path(img_path).relative_to(self.image_dir).with_suffix(".txt")
        )

        # ---- Загрузка изображения ----
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([image] * 3, axis=-1)
        # image = cv2.resize(image, self.image_size)
        h, w = image.shape[:2]

        # ---- Чтение keypoints ----
        with open(label_path, 'r') as f:
            data = list(map(float, f.readline().strip().split()))
        
        kps = data[5:]  # пропускаем bbox
        keypoints = []
        for i in range(self.num_keypoints):
            # breakpoint()
            # print(kps)
            # try:
            x = kps[i * 3]
            # except:
            #     breakpoint()
            y = kps[i * 3 + 1]
            keypoints.append((x, y))  # нормализованные [0..1]

        # ---- Albumentations работает только с пикселями ----
        # breakpoint()
        kp_pixel = [(int(x * w), int(y * h)) for x, y in keypoints]

        # ---- Применяем albumentations ----
        if self.transform:
            transformed = self.transform(image=image, keypoints=kp_pixel)
            image = transformed['image'] / 255
            kp_transformed = transformed['keypoints']

            # Переводим в тензор
            # image_tensor = transforms.ToTensor()(image_transformed)
            # image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

            # Переводим обратно в нормализованный вид
            # kp_normalized = [(x / w, y / h) for x, y in kp_transformed]
            keypoints_tensor = torch.tensor(kp_transformed, dtype=torch.float32)
        else:
            # Без аугментаций — просто нормализуем
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        return image, keypoints_tensor.view(-1, 2), 1.0
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    hard_sample_manager = HardSampleManager(top_percent=0.2, repeat_times=3, loss_weight=2.0)

    # Параметры
    base_dir = '/media/4TB/HAGRID/hagridv2_512/crops'
    image_dir_train = f"{base_dir}/train_krasota/images"
    label_dir_train = f"{base_dir}/train_krasota/labels"
    image_dir_val = f"{base_dir}/val/images"
    label_dir_val = f"{base_dir}/val/labels"
    
    image_hagrid_dir_train = f"{base_dir}/hagrid_only/images"
    label_hagrid_dir_train = f"{base_dir}/hagrid_only/labels"

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    transform_hagrid = A.Compose([
        A.Resize(height=30, width=30),
        A.Resize(height=256, width=256),
        # A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # train_dataset_hands = HandKeypointsDataset(image_dir=image_dir_train, label_dir=label_dir_train, transform=transform)
    val_dataset = HandKeypointsDataset(image_dir=image_dir_val, label_dir=label_dir_val, transform=transform)
    # train_hagrid_dataset = HandKeypointsDataset(image_dir=image_hagrid_dir_train, label_dir=label_hagrid_dir_train, transform=transform_hagrid)
    # train_dataset = ConcatDataset([train_dataset_hands, train_hagrid_dataset])
    train_dataset = HandKeypointsDataset(image_dir=image_dir_train, label_dir=label_dir_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ---- Инициализация модели ----
    # model = HandKeypointNet(num_keypoints=21).to('cuda')
    model = BlazeHandLandmark()
    state_dict = torch.load("blazehand_landmark.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to('cuda')
    # breakpoint()
    # ckpt = torch.load('ckpts2/best_hand_kp_model_11_967.7938973135199.pth')
    # model.load_state_dict(ckpt)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    # ckpt = torch.load('best2.pth')
    # model.load_state_dict(ckpt)
    # breakpoint()
    # pred = model(train_dataset[0][0].unsqueeze(0))
    # ---- Логирование ----
    writer = SummaryWriter(log_dir="runs/hand_keypoint_net")
    best_val_loss = float('inf')
    import shutil
    shutil.rmtree('results')
    os.makedirs('results')
    # ---- Цикл обучения ----
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Всего параметров: {total:,}")
        print(f"Обучаемые: {trainable:,}")
        print(f"Замороженные: {frozen:,}")

    x = count_parameters(model)
    for epoch in trange(200):  # например, 50 эпох
        if epoch == 0:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        else:
            # ---- Иначе создаём датасет с повторами hardest примеров ----
            train_dataset_with_hard = hard_sample_manager.get_dataset(train_dataset)
            train_loader = DataLoader(train_dataset_with_hard, batch_size=32, shuffle=True)
        model.train()
        train_loss = 0
        all_losses = []
        all_images = []
        all_keypoints = []
        # На каждой эпохе создаём датасет с hardest examples

        for batch_idx, (images, keypoints, weights) in enumerate(tqdm(train_loader, desc=f'{train_loss}')):
            
            images = images.to('cuda')
            keypoints = keypoints.to('cuda')
            weights = weights.to('cuda')
            # breakpoint()
            # print(keypoints)
            # breakpoint()

            optimizer.zero_grad()
            keypoints_preds = model(images)[:,:,:2] * 256 # [B, 21, 2]
            # breakpoint()
            per_sample_loss = ((keypoints_preds - keypoints) ** 2).mean(dim=(1, 2))  # [B]
            weighted_loss = (per_sample_loss * weights).mean()
            loss = weighted_loss
            # loss = criterion(keypoints_preds, keypoints) * 5
            # breakpoint()
            # loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()

            # ---- Сохраняем данные для HNM ----
            with torch.no_grad():
                all_losses.extend(per_sample_loss.cpu().tolist())
                all_images.extend(images.cpu())
                all_keypoints.extend(keypoints.cpu())

           

            # train_loss += loss.item()
            # torch.save(model.state_dict(), "best_hand_kp_model.pth")
            print(f'LOSS: {loss.item()}, EPOCH: {epoch}')
            # ---- Визуализация ----
            if batch_idx % 100 == 0:
                with torch.no_grad():
                        indices = random.sample(range(len(images)), k=2)
                        selected_images = images[indices].cpu().numpy()
                        selected_preds = keypoints_preds[indices].cpu().numpy()
                        selected_gt = keypoints[indices].cpu().numpy()

                        for i, idx in enumerate(indices):
                            img = selected_images[i].transpose(1, 2, 0)  # CHW -> HWC
                            pred_kps = selected_preds[i].reshape(21, 2)
                            true_kps = selected_gt[i].reshape(21, 2)    
                
# Умножаем нормализованные координаты на размер изображения
                            h, w = img.shape[0], img.shape[1]
                            pred_pixels = [(int(x), int(y)) for x, y in pred_kps]
                            true_pixels = [(int(x), int(y)) for x, y in true_kps]

                            # Конвертируем в uint8 для отображения
                            img_to_draw = (img * 255).astype(np.uint8).copy()
                            img_gt = img_to_draw.copy()
                            img_pred = img_to_draw.copy()

                            # Рисуем точки
                            for (x, y) in true_pixels:
                                cv2.circle(img_gt, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # зелёные точки

                            for (x, y) in pred_pixels:
                                cv2.circle(img_pred, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # красные точки

                            # Соединяем ground truth и предсказание на одном изображении
                            combined = cv2.addWeighted(img_gt, 0.5, img_pred, 0.5, 0)

                            # Сохраняем
                            cv2.imwrite(f"results/epoch_{epoch}_batch_{batch_idx}_sample_{idx}.jpg", combined[:, :, ::-1])  # RGB -> BGR

        # ---- Валидация ----
        hard_sample_manager.update(all_losses, all_images, all_keypoints)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            j = 0
            for images, keypoints, _ in tqdm(val_loader):
                images = images.to('cuda')
                keypoints = keypoints.to('cuda')
                keypoints_pred = model(images)
                loss = criterion(keypoints_pred, keypoints)
                val_loss += loss.item()
                j += 1
                if j == 10:
                    break

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}"                                                                                                                                                                                                                                                                                                                                                                                                         ) #| Val Loss: {avg_val_loss:.4f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        # writer.add_scalar("Loss/val", avg_val_loss, epoch)
        torch.save(model.state_dict(), f"ckpts12/best_hand_kp_model_{epoch}_{avg_train_loss}.pth")
        # ---- Сохранение лучшей модели ----
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), "best_hand_kp_model.pth")
        #     print(f"Модель сохранена как best_hand_kp_model.pth (Val Loss: {avg_val_loss:.4f})")
