from ultralytics import YOLO
import time
import numpy as np
import pyvidu as vidu
import cv2
import torch
from collections import defaultdict
from GMHD_osc_sender import Sender
import timm
from torchvision import transforms
from PIL import Image
from new_epoch_coord import CoordinateTransformer #pixel_to_world_3d, pixel_to_world
# from train_kps import BlazeHandLandmark
from run_trt import load_hand_gesture_model, predict_hand_gesture
from new_speed_bench import run_model_batch
# from coordinate_transformer import CoordinateTransformer

# def pixel_to_world(u, v, K, dist, H):
#         pixel = np.array([[u, v]], dtype=np.float64)
#         pixel = pixel.reshape(-1,1,2)
#         pixel = cv2.undistortPoints(pixel, K, dist, P=K)
#         # pixel = pixel.reshape(-1,1,2)
#         x_norm, y_norm = pixel[0][0]
#         pixel = np.array([x_norm, y_norm, 1.0])
#         world = np.linalg.inv(H) @ pixel
#         world /= world[2]  # нормализация
#         X, Y = world[0], world[1]
#         return X, Y

# def to_world(u, v):
#     import numpy as np
#     import cv2

#     # ==== ВХОДНЫЕ ДАННЫЕ ====
#     # матрица внутренних параметров камеры (пример, замени своими)
#     K = np.array([[529.141724, 0, 329.699677],
#                 [0, 528.957397, 229.188171],
#                 [0, 0, 1]], dtype=np.float64)

#     # дисторсия (если знаешь, замени своими)
#     dist = np.array([-0.2333385, 0.102841, -0.017275, 0.000101, 0.000010])

#     # высота камеры над столом
#     height = 182.0  # мм

#     # угол наклона камеры (в радианах)
#     alpha = np.deg2rad(21)

#     # ==== ПОЗИЦИЯ КАМЕРЫ ====
#     # камера смотрит вниз, повёрнута вокруг оси X на угол alpha
#     R_down = np.array([[1, 0, 0],
#                     [0, np.cos(alpha), -np.sin(alpha)],
#                     [0, np.sin(alpha),  np.cos(alpha)]], dtype=np.float64)

#     t = np.array([[0], [0], [height]])  # камера на высоте height

#     # Матрица проекции
#     RT = np.hstack((R_down, t))

#     # ==== СЧИТАЕМ ГОМОГРАФИЮ (для плоскости Z=0) ====
#     # берем первые 3 столбца (X,Y,Z=0 → значит берем только X,Y)
#     H = K @ RT[:, [0,1,3]]

#     # ==== ФУНКЦИЯ: пиксель → мировые координаты (X,Y) ====
    

#     # ==== ПРИМЕР ====
#     # u, v = 700, 400  # пиксель на изображении
#     X, Y = pixel_to_world(u, v, K, dist, H)
#     print(f"Pixel ({u},{v}) -> World ({X:.2f}, {Y:.2f}) мм")
#     return X, Y
    

coord_transformer = CoordinateTransformer()
# from iou_tracker import KalmanIOUTracker

# Инициализация OSC отправителя
# sender = Sender(ip="127.0.0.1", port=5055)
# sender = Sender(ip="192.168.50.209", port=5055, logging_level="DEBUG")
sender = Sender(ip="10.0.0.101", port=5055, logging_level="DEBUG")

# Загрузка модели YOLO
model = YOLO("/home/cineai/ViduSdk/python/TRT_Roma/newone/upside.engine")
trt_model = load_hand_gesture_model("my_model.engine")

# Создание словаря для хранения предыдущих детекций каждого трека
previous_detections = defaultdict(list)
kps_model = load_hand_gesture_model("cls_kps.engine")

# Функция для сглаживания детекций
def smooth_detection(track_id, current_detection):
    # Добавляем текущую детекцию для этого трека
    previous_detections[track_id].append(current_detection)

    # Ограничиваем количество сохранённых детекций
    if len(previous_detections[track_id]) > 2:
        previous_detections[track_id].pop(0)

    # Сглаживание: берём среднее значение последних 2 детекций
    smoothed_box = np.mean(previous_detections[track_id], axis=0).astype(int)
    return smoothed_box

# Функция для детектирования рук в видео
def detect_hands_in_video():
    # breakpoint()
    device = vidu.PDdevice()
    if not device.init():
        print("GENERAL ERROR")
        exit(-1)

    # num_classes = 5
    # model_hands_cls = timm.create_model('fastvit_sa24', pretrained=True, num_classes=5)
    # # model_hands_cls.fc = nn.Linear(model_hands_cls.fc.in_features, num_classes)
    # transform_hands_cls = transforms.Compose([
    #     transforms.Resize((400, 400)),  # Изменяем размер изображений
    #     # transforms.RandomHorizontalFlip(),  # Горизонтальное отражение
    #     # transforms.RandomRotation(10),     # Поворот изображения
    #     transforms.ToTensor(),             # Преобразование в тензор
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    # ])
    # ckpts_hands_cls = torch.load('/home/cinepost_jet/Vid_v2/hand_gesture_classifier.pth')
    # model_hands_cls.load_state_dict(ckpts_hands_cls)
    # model_hands_cls.to('cuda')
    # model_hands_cls.eval()
    # class_hands_cls = {'2_hands': 0, 'fist': 1, 'normal_hand': 2, 'pinch': 3, 'pointer': 4}
    # class_hands_cls = {v: k for k, v in class_hands_cls.items()}

    # model_kps = BlazeHandLandmark().to("cuda")
    # model_kps.load_state_dict(torch.load("/home/cinepost_jet/Vid_v2/best_hand_kp_model.pth"))  # загрузка весов
    # model_kps.eval()%s: %i" % (streamname, 0)
    streamnum = device.getStreamNum()
    CROP_SIZE = 256
    # tracker = KalmanIOUTracker(max_age=3, iou_threshold=0.3)
    # trt_model = load_hand_gesture_model("/home/cinepost_jet/Vid_v2/TRT_Roma/mobilenetv3_hand_cls.engine")
    with vidu.PDstream(device, 1) as stream:
        stream.init()
        streamname = stream.getStreamName()
        print("Stream =", streamname)
        image = np.zeros((0, 0, 0), dtype="uint8")

        if streamname == "ToF":
            # Настройки ToF-камеры
            stream.set("ToF::StreamFps", 100)
            stream.set("ToF::Exposure", 0.15)
            stream.set("ToF::Distance", 7.5)
            stream.set("ToF::DepthMedianBlur", 0)
            stream.set("ToF::DepthFlyingPixelRemoval", 2)
            stream.set("ToF::Threshold", 40)
            stream.set("ToF::Gain", 9)
            stream.set("ToF::AutoExposure", 0)
            stream.set("ToF::DepthSmoothStrength", 0)
            stream.set("ToF::DepthCompletion", 0)

            # Инициализация FPS
            previousTime_FPS = 0
            currentTime_FPS = 0
            startTime = time.time()
            jjj = 0
            while True:
                images = stream.getPyImage()
                if len(images) == 2:
                    
                    intrinsic = vidu.intrinsics()
                    extrinsic = vidu.extrinsics()
                    stream.getCamPara(intrinsic, extrinsic)
                    # vidu.print_intrinsics(intrinsic)
                    # Преобразование изображения
                    image = images[1]
                    depth = images[0]
                    cv2.imwrite('temp.png', depth)
                    # image = cv2.flip(image, -1)
                    # depth = cv2.flip(image, -1)
                    
                    ir_image_8bit = cv2.convertScaleAbs(image, alpha=(2000.0 / 60000.0))
                    _, ir_image_8bit = cv2.threshold(ir_image_8bit, 10, 60, cv2.THRESH_TOZERO)
                    image = np.copy(ir_image_8bit)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    draw_image = image.copy()
                    jjj += 1
                    if jjj % 8 != 0:
                        continue
                    image_h, image_w = image.shape[:2]
                    # Инференс с трекингом
                    results = model.predict(image)

                    draw_image = image.copy()
                    list_src_crops, resized_crops = [], []
                    detections = []
                    for i, box in enumerate(results[0].boxes.xyxy[:2]):
                        x1, y1, x2, y2 = map(int, box.tolist())
                    #     detections.append([x1,y1,x2,y2])
                    # tracked = tracker.update(detections)
                    # for track_id, x1, y1, x2, y2 in tracked:
                        cls = int(results[0].boxes.cls[i].item())
                        cropped_hand = image[y1:y2, x1:x2]
                        list_src_crops.append(cropped_hand)
                        # cropped_resized = cv2.resize(cropped_hand, (CROP_SIZE, CROP_SIZE))
                        #cropped_resized = cv2.resize(cv2.resize(cropped_hand, (80,80)), (CROP_SIZE, CROP_SIZE))
                        cropped_resized = cv2.resize(cropped_hand, (CROP_SIZE, CROP_SIZE))
                        cv2.imwrite('temp.png', cropped_resized)
                        resized_crops.append(cropped_resized)
                        # # input_tensor = torch.from_numpy(cropped_resized).permute(2,0,1).unsqueeze(0).to("cuda") / 255
                        # # with torch.no_grad():
                        # #     # breakpoint()
                        # #     output = model(input_tensor)[:,:,:2] * 256
                        # # preds = output.squeeze(0).cpu().numpy()  # [21, 2]
                        # # for p_id, (x_norm, y_norm) in enumerate(preds):
                        # #     x_norm, y_norm = x_norm / CROP_SIZE, y_norm / CROP_SIZE
                        # #     px = int(x_norm * image_w)
                        # #     py = int(y_norm * image_h)
                        # #     abs_px = x1 + px
                        # #     abs_py = y1 + py
                        # #     cv2.circle(draw_image, (abs_px, abs_py), 1, (0,255,0), -1)
                        # #     cropped_hand = image[y1:y2, x1:x2]
                        # #     cropped_hand640 = cv2.resize(cropped_hand, (640,640))
                        # input_tensor_cls = transform_hands_cls(Image.fromarray(cropped_hand640)).unsqueeze(0).float().to(device)
                        # with torch.no_grad():
                        #     output = model_hands_cls(input_tensor_cls)
                        #     # breakpoint()
                        #     _, predicted = torch.max(output, 1)
                        #     # breakpoint()
                        #     predicted_class = class_hands_cls[predicted.item()]
                        map_colors = {0: (255,0,0), 1: (255,128,0), 2: (102,255,255), 3: (153,0,153), 4: (178,102,55)}
                        # # if predicted_class != 'pinch':
                            
                        # color = map_colors[predicted.item()]
                        # Только руки или нужный класс (измени при необходимости)
                        
                        # cls_class, conf = predict_hand_gesture(cropped_resized, trt_model)
                        # color = map_colors[0]
                        
                        # HAND_CLASS_ID = 0
                        # if cls != HAND_CLASS_ID:
                        #     continue

                        # track_id = results[0].boxes.id.int().tolist()[i] if results[0].boxes.id is not None else -1
                        # if track_id == -1:
                        #     continue  # пропуск без ID

                        # Сохраняем текущую детекцию
                        current_detection = [x1, y1, x2, y2]

                        # Сглаживание детекции
                        # smoothed_box = smooth_detection(track_id, current_detection)
                        # x1_smooth, y1_smooth, x2_smooth, y2_smooth = smoothed_box

                        # Расширяем бокс на 1%
                        x1_smooth, y1_smooth, x2_smooth, y2_smooth = x1, y1, x2, y2
                        expand_width = int(0.01 * (x2_smooth - x1_smooth))
                        expand_height = int(0.01 * (y2_smooth - y1_smooth))

                        x1_smooth = max(0, x1_smooth - expand_width)
                        y1_smooth = max(0, y1_smooth - expand_height)
                        x2_smooth += expand_width
                        y2_smooth += expand_height


                        z = float(0)
                        pt_tl = [float(x1_smooth),float(y1_smooth),z]
                        pt_tr = [float(x2_smooth),float(y1_smooth),z]
                        pt_bl = [float(x1_smooth),float(y2_smooth),z]
                        pt_br = [float(x2_smooth),float(y2_smooth),z]

                        vctl = np.array(pt_tl)
                        vcbr = np.array(pt_br)
                        pt_ct = (vcbr+vctl)/2                   

                        pts = (pt_ct,pt_tl,pt_tr,pt_bl,pt_br)
                        for j,pt in enumerate(pts):
                            print(j,pt[0],pt[1],pt[2])
                            sender.send(address=f"/bboxes/bbox_{i}/point_{j}", data=[pt[0],pt[1],pt[2]])
                        # Рисуем bounding box и ID
                        # color = (0, 255, 0)
                        track_id = 0
                        # cv2.rectangle(draw_image, (x1_smooth, y1_smooth), (x2_smooth, y2_smooth), (0,0,255), 2)
                        # cv2.putText(draw_image, f'id:{track_id}', (x1_smooth, y1_smooth - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if len(list_src_crops) == 0:
                        continue
                    with torch.no_grad():
                        # breakpoint()
                        output = run_model_batch(resized_crops, kps_model, image_size=256)
                    kps_preds = output[0]
                    

                    cls_output = run_model_batch(resized_crops, trt_model, image_size=256)[0]

                    for i, box in enumerate(results[0].boxes.xyxy[:2]):
                    # for track_id, x1, y1, x2, y2 in tracked:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        label = cls_output[i].argmax()
                        color = map_colors[label]
                        preds = np.expand_dims(kps_preds[i], 0)[:,:,:2] * 256
                        preds = preds[0]
                        h, w, _ = list_src_crops[i].shape
                        points = []
                        for p_id, (x_norm, y_norm) in enumerate(preds):
                            x_norm, y_norm = x_norm / CROP_SIZE, y_norm / CROP_SIZE
                            px = int(x_norm * w)
                            py = int(y_norm * h)
                            abs_px = x1 + px
                            abs_py = y1 + py
                            if p_id == 0:
                                my_x, my_y = abs_px, abs_py
                                z = depth[my_y, my_x] * (7.5/65536)
                                cv2.circle(draw_image, (my_x,my_y), 1, (0,255,0), -1)
                                my_x_normal = 640 - 1 -my_x
                                my_y_normal = 480 -1 - my_y
                                # real_x, real_y = coord_transformer.pixel_to_world(my_x, my_y, z)
                                real_x, real_y, real_z = coord_transformer.pixel_to_floor_3d(my_x_normal, my_y_normal, z)
                                #real_z -= 0.8
                                cv2.putText(draw_image, f'{real_x:.3f} {real_y:.3f}, {real_z:.3f}, {z:.3f}', (my_x,my_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            points.append((abs_px, abs_py))
                            if p_id == 8 or p_id == 4:
                                if p_id == 8:
                                    p8 = px, py
                                if p_id == 4:
                                    p4 = px, py
                                color = (117, 0, 178)
                            cv2.circle(draw_image, (abs_px, abs_py), 1, (0,255,0), -1)  # красные точки
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
                            cv2.line(draw_image, start, end, (0,255,0), thickness=1)
                                        # Отображение изображения
                        cv2.line(draw_image, (0, 550), (550, 550), color, 2)

                        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(draw_image, f'id:{track_id}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow("%s: %i" % (streamname, 0), draw_image)
                    cv2.imshow("kek", depth)
                    cv2.imwrite(f'roma_images/{jjj}.png', draw_image)
                    # Обработка клавиш
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

# Запуск детектора
detect_hands_in_video()
