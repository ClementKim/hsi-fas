import os
import cv2
import numpy as np
from ultralytics import YOLO

# 설정
input_root = 'preprocessing/face_crop/image_to_crop'
output_root = 'preprocessing/face_crop/output'
model_path = 'preprocessing/face_crop/yolov8-face/yolov8n-face.pt'
output_size = 256
padding_ratio = 0.7  # 얼굴 주변 얼마나 여유 줄지

# YOLOv8-face 모델 로드
model = YOLO(model_path)
last_face_box = None
prev_folder = None

# 이미지 순회
for subdir, _, files in os.walk(input_root):
    for file in sorted(files):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_path = os.path.join(subdir, file)
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = cv2.imread(input_path)
        h_img, w_img = img.shape[:2]

        current_folder = os.path.dirname(relative_path)
        if current_folder != prev_folder:
            last_face_box = None
            prev_folder = current_folder

        # 얼굴 탐지
        results = model.predict(source=img, conf=0.3, iou=0.45, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []

        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            last_face_box = (x1, y1, x2, y2)
        elif last_face_box:
            print(f"[!] 얼굴 없음: {relative_path} → 이전 박스 재사용")
            x1, y1, x2, y2 = last_face_box
        else:
            print(f"[!] 얼굴 없음 & 이전 박스도 없음: {relative_path} → 중앙 fallback")
            cx, cy = w_img // 2, h_img // 2
            square_half = min(w_img, h_img) // 3
            x1, y1 = cx - square_half, cy - square_half
            x2, y2 = cx + square_half, cy + square_half

        # 얼굴 중심 기준 정사각형 crop 영역 계산
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * padding_ratio), int(h * padding_ratio)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        square_half = int(max(w + 2 * pad_w, h + 2 * pad_h) / 2)

        # 이미지 경계 고려 정사각형 crop 좌표 조정
        left = max(cx - square_half, 0)
        top = max(cy - square_half, 0)
        right = min(cx + square_half, w_img)
        bottom = min(cy + square_half, h_img)

        crop_width = right - left
        crop_height = bottom - top
        side = min(crop_width, crop_height)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        new_left = max(center_x - side // 2, 0)
        new_top = max(center_y - side // 2, 0)
        new_right = new_left + side
        new_bottom = new_top + side

        # 이미지 자르기 + 256x256 리사이즈
        cropped = img[new_top:new_bottom, new_left:new_right]
        resized = cv2.resize(cropped, (output_size, output_size))
        cv2.imwrite(output_path, resized)
