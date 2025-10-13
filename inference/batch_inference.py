"""
Batch inference script for hand detection, keypoint estimation, and gesture classification.
Supports tracking and visualization.
"""

import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import timm


class KalmanIOUTracker:
    """Simple IOU-based tracker with Kalman filtering."""
    
    def __init__(self, max_age=3, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0

    def compute_iou(self, box1, box2):
        """Compute IOU between two boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / float(box1_area + box2_area - intersection)
        return iou

    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections: List of [x1, y1, x2, y2] boxes
            
        Returns:
            List of (track_id, x1, y1, x2, y2) tuples
        """
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in unmatched_detections:
                iou = self.compute_iou(track_data['box'], detections[det_idx])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Matched
                self.tracks[track_id]['box'] = detections[best_det_idx]
                self.tracks[track_id]['age'] = 0
                matched_tracks.append((track_id, *detections[best_det_idx]))
                unmatched_detections.remove(best_det_idx)
            else:
                # No match, age the track
                self.tracks[track_id]['age'] += 1
        
        # Remove old tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] < self.max_age}
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                'box': detections[det_idx],
                'age': 0
            }
            matched_tracks.append((track_id, *detections[det_idx]))
        
        return matched_tracks


def draw_skeleton(image, keypoints, connections=None):
    """Draw hand skeleton on image.
    
    Args:
        image: Image to draw on
        keypoints: List of (x, y) tuples
        connections: List of (start_idx, end_idx) tuples
    """
    if connections is None:
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),
            (13,14),(14,15),(15,16),
            (17,18),(18,19),(19,20),
            (0,5),(5,9),(9,13),(13,17),(0,17)
        ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = tuple(map(int, keypoints[start_idx]))
            end = tuple(map(int, keypoints[end_idx]))
            cv2.line(image, start, end, (0, 255, 0), thickness=2)
    
    # Draw keypoints
    for kp in keypoints:
        cv2.circle(image, tuple(map(int, kp)), 3, (0, 0, 255), -1)
    
    return image


def run_inference(args):
    """Run inference on image directory."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load YOLO model for hand detection
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_model)
    
    # Load keypoint/gesture model
    if args.model_type == 'blazehand':
        from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark
        model = BlazeHandLandmark()
    elif args.model_type == 'timm':
        model = timm.create_model(
            args.timm_model,
            pretrained=False,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()
    
    # Setup tracker
    tracker = KalmanIOUTracker(max_age=args.track_max_age, iou_threshold=args.track_iou_threshold)
    
    # Gesture class names
    class_names = {0: '2_hands', 1: 'fist', 2: 'normal_hand', 3: 'pinch', 4: 'pointer'}
    color_map = {0: (255,0,0), 1: (255,128,0), 2: (102,255,255), 3: (153,0,153), 4: (178,102,55)}
    
    # Get image files
    image_files = glob(os.path.join(args.input_dir, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    for img_idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        if args.flip:
            image = cv2.flip(image, -1)
        
        vis_image = image.copy()
        
        # Detect hands with YOLO
        results = yolo_model(image)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue
        
        # Prepare detections for tracker
        detections = []
        for box in results[0].boxes.xyxy[:args.max_hands]:
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Expand box
            box_width = x2 - x1
            box_height = y2 - y1
            expand_w = int(args.expand_ratio * box_width)
            expand_h = int(args.expand_ratio * box_height)
            
            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(image.shape[1], x2 + expand_w)
            y2 = min(image.shape[0], y2 + expand_h)
            
            detections.append([x1, y1, x2, y2])
        
        # Update tracker
        tracked = tracker.update(detections)
        
        # Process each tracked hand
        for track_id, x1, y1, x2, y2 in tracked:
            # Crop hand
            cropped_hand = image[y1:y2, x1:x2]
            
            if cropped_hand.size == 0:
                continue
            
            # Prepare input
            cropped_resized = cv2.resize(cropped_hand, (args.img_size, args.img_size))
            
            if args.grayscale:
                gray_crop = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
                cropped_resized = np.stack([gray_crop] * 3, axis=-1)
            
            input_tensor = torch.from_numpy(cropped_resized).permute(2, 0, 1).unsqueeze(0).to(device)
            input_tensor = input_tensor.float() / 255.0
            
            # Run model
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    # Multitask: (classification, keypoints)
                    cls_output, kps_output = outputs
                    
                    # Get gesture class
                    cls_label = cls_output.argmax(dim=1).item()
                    gesture_name = class_names.get(cls_label, f"class_{cls_label}")
                    color = color_map.get(cls_label, (255, 255, 255))
                    
                    # Get keypoints
                    keypoints_pred = kps_output[0, :, :2].cpu().numpy() * args.img_size
                else:
                    # Keypoints only
                    keypoints_pred = outputs[0, :, :2].cpu().numpy() * args.img_size
                    gesture_name = "hand"
                    color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {gesture_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Draw keypoints
            if 'keypoints_pred' in locals():
                h_crop, w_crop = cropped_hand.shape[:2]
                abs_keypoints = []
                
                for kp in keypoints_pred:
                    x_norm, y_norm = kp[0] / args.img_size, kp[1] / args.img_size
                    px = int(x1 + x_norm * w_crop)
                    py = int(y1 + y_norm * h_crop)
                    abs_keypoints.append((px, py))
                
                if args.draw_skeleton:
                    draw_skeleton(vis_image, abs_keypoints)
        
        # Save result
        output_path = os.path.join(args.output_dir, f"{img_idx:06d}.jpg")
        cv2.imwrite(output_path, vis_image)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch inference for hand detection and gesture recognition')
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results')
    
    # Models
    parser.add_argument('--yolo_model', type=str, required=True,
                        help='Path to YOLO model for hand detection')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to gesture/keypoint model weights')
    parser.add_argument('--model_type', type=str, default='blazehand',
                        choices=['blazehand', 'timm'])
    parser.add_argument('--timm_model', type=str, default='mobilenetv3_small_100',
                        help='Timm model name (if model_type=timm)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of gesture classes')
    
    # Processing
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input size for gesture model')
    parser.add_argument('--max_hands', type=int, default=4,
                        help='Maximum number of hands to process per image')
    parser.add_argument('--expand_ratio', type=float, default=0.3,
                        help='Ratio to expand detection boxes')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert to grayscale')
    parser.add_argument('--flip', action='store_true',
                        help='Flip images')
    
    # Tracking
    parser.add_argument('--track_max_age', type=int, default=3,
                        help='Maximum age for tracks')
    parser.add_argument('--track_iou_threshold', type=float, default=0.3,
                        help='IOU threshold for matching tracks')
    
    # Visualization
    parser.add_argument('--draw_skeleton', action='store_true',
                        help='Draw hand skeleton')
    
    args = parser.parse_args()
    run_inference(args)

