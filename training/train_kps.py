
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
from train_cls import FocalLoss, accuracy
import shutil
import timm
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
# from MediaPipePyTorch.blazebase import BlazeLandmark, BlazeBlock
from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark

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
from clearml import Task


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
        cls_id = int(data[0])
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

        return image, cls_id, keypoints_tensor.view(-1, 2)
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # Параметры
    shutil.rmtree('results')
    os.makedirs('results')
    task = Task.init(
        project_name="hand_kps_and_gesture",
        task_name="train_multitask",
        output_uri=True  # сохранение артефактов
    )
    logger = task.get_logger()
    base_dir = '/media/4TB/HAGRID/hagridv2_512/crops'
    image_dir_train = f"{base_dir}/train_multitask/images"
    label_dir_train = f"{base_dir}/train_multitask/labels"
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
    
    
    
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    # --- 3. Убеждаемся, что классы 0..4 ---
    assert train_labels.min() == 0 and train_labels.max() == 4, "cls_id должны быть 0,1,2,3,4!"

    # --- 4. Вычисляем веса ---
    cls_counts = np.bincount(train_labels, minlength=5)
    class_weights = len(train_labels) / (len(cls_counts) * cls_counts)
    class_weights = torch.FloatTensor(class_weights).to('cuda')

    print("Class weights:", class_weights.cpu().numpy())

    # --- 5. Создаём WeightedRandomSampler ---
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False,  sampler=sampler,)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ---- Инициализация модели ----
    # model = HandKeypointNet(num_keypoints=21).to('cuda')
    model = BlazeHandLandmark()
    state_dict = torch.load("blazehand_landmark.pth")  #blazehand_landmark.pth
    model.load_state_dict(state_dict, strict=False)
    model.to('cuda')
    # breakpoint()
    # ckpt = torch.load('ckpts2/best_hand_kp_model_11_967.7938973135199.pth')
    # model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    # cls_criterion = FocalLoss()
    weights = torch.tensor([2.0, 2.0, 1.0, 2.0, 8.0]).to('cuda')
    cls_criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1  # ! Очень важно при дисбалансе
    )
    # ckpt = torch.load('best2.pth')
    # model.load_state_dict(ckpt)
    # breakpoint()
    # pred = model(train_dataset[0][0].unsqueeze(0))
    # ---- Логирование ----
    writer = SummaryWriter(log_dir="runs/hand_keypoint_net")
    best_val_loss = float('inf')

    # ---- Цикл обучения ----
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Всего параметров: {total:,}")
        print(f"Обучаемые: {trainable:,}")
        print(f"Замороженные: {frozen:,}")
    train_accuracies = []
    epochs_list = []
    x = count_parameters(model)
    for epoch in trange(200):  # например, 50 эпох
        epochs_list.append(epoch + 1)
        model.train()
        train_acc = 0.0
        train_loss = 0

        for batch_idx, (images, cls_id, keypoints) in enumerate(tqdm(train_loader, desc=f'{train_loss}')):
            
            images = images.to('cuda')
            keypoints = keypoints.to('cuda')
            cls_id = cls_id.to('cuda')
            # breakpoint()
            # print(keypoints)
            # breakpoint()

            optimizer.zero_grad()
            cls_pred, keypoints_preds = model(images)#[:,:,:2] * 256 # [B, 21, 2]
            keypoints_preds = keypoints_preds[:,:,:2] * 256
            # breakpoint()
            loss = criterion(keypoints_preds, keypoints) * 1 + cls_criterion(cls_pred, cls_id) * 2
            # breakpoint()
            # loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            train_acc += accuracy(cls_pred, cls_id)

            train_loss += loss.item()
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
        train_acc /= len(train_loader)
        train_accuracies.append(train_acc)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, train_accuracies, label='Train Accuracy', marker='o', linewidth=2)
        plt.title('Training Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Сохраняем график как изображение
        plt.savefig(f"acc_results/accuracy_epoch_{epoch+1}.png")
        plt.close()  # обязательно закрываем, чтобы не течь память!
        logger.report_scalar("Accuracy", "Train", iteration=epoch + 1, value=train_acc)
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     j = 0
        #     for images, keypoints in tqdm(val_loader):
        #         images = images.to('cuda')
        #         keypoints = keypoints.to('cuda')
        #         cls_pred, keypoints_pred = model(images)
        #         loss = criterion(keypoints_pred, keypoints)
        #         val_loss += loss.item()
        #         j += 1
        #         if j == 10:
        #             break

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}"                                                                                                                                                                                                                                                                                                                                                                                                         ) #| Val Loss: {avg_val_loss:.4f}")

        # writer.add_scalar("Loss/train", avg_train_loss, epoch)
        # writer.add_scalar("Loss/val", avg_val_loss, epoch)
        torch.save(model.state_dict(), f"ckpts13_pointer_krasota/best_hand_kp_model_{epoch}_{avg_train_loss}.pth")
        # ---- Сохранение лучшей модели ----.
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss. 
        #     torch.save(model.state_dict(), "best_hand_kp_model.pth")
        #     print(f"Модель сохранена как best_hand_kp_model.pth (Val Loss: {avg_val_loss:.4f})")
