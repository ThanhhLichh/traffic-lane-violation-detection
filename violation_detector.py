# violation_detector_fixed.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cdist

# === Load mô hình YOLOv8 ===
model = YOLO("yolov8m.pt")

# === Load vùng làn đã lưu ===
car_lane = np.load("lanes/car_lane.npy")
motorcycle_lane = np.load("lanes/motorcycle_lane.npy")

# === Khởi tạo DeepSORT tracker ===
tracker = DeepSort(max_age=5)

# === Tạo thư mục lưu ảnh xe vi phạm ===
os.makedirs('violated_vehicles/car', exist_ok=True)
os.makedirs('violated_vehicles/motorcycle', exist_ok=True)

# === Hàm kiểm tra 1 điểm có nằm trong vùng đa giác hay không ===
def is_inside(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# === Gom box gần nhau bằng khoảng cách tâm box ===
def merge_nearby_boxes(detections, distance_thresh=50):
    if not detections:
        return []

    centers = np.array([[x + w/2, y + h/2] for (x, y, w, h), _, _ in detections])
    merged = []
    used = set()

    for i in range(len(detections)):
        if i in used:
            continue
        x1, y1, w1, h1 = detections[i][0]
        conf1, label1 = detections[i][1], detections[i][2]

        group = [(x1, y1, w1, h1)]
        used.add(i)

        for j in range(i + 1, len(detections)):
            if j in used:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_thresh:
                x2, y2, w2, h2 = detections[j][0]
                group.append((x2, y2, w2, h2))
                used.add(j)

        xs = [g[0] for g in group]
        ys = [g[1] for g in group]
        ws = [g[2] for g in group]
        hs = [g[3] for g in group]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max([x + w for x, w in zip(xs, ws)])
        y_max = max([y + h for y, h in zip(ys, hs)])
        merged.append(([x_min, y_min, x_max - x_min, y_max - y_min], conf1, label1))

    return merged

# === Load video ===
cap = cv2.VideoCapture("duong_pho4.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Thiết lập VideoWriter để lưu output video ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
out = cv2.VideoWriter('output_violation.mp4', fourcc, fps, (width, height))


violated_ids = set()
track_id_to_label = {}
output_txt = open("violations.txt", "w")

colors = {
    'car': (255, 0, 0),
    'motorcycle': (0, 255, 0),
    'violation': (0, 0, 255)
}

max_width = width * 0.6
max_height = height * 0.6
max_area = width * height * 0.5
min_area = 2000

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = model.names[cls_id]

        if label not in ['car', 'motorcycle']:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        if w > max_width or h > max_height:
            continue
        if area > max_area or area < min_area:
            continue
        if aspect_ratio > 2.5 or aspect_ratio < 0.2:
            continue
        if h < 30 or w < 30:
            continue
        if h > w * 2:
            continue

        detections.append(([x1, y1, w, h], conf, label))

    # Gom các box gần nhau lại
    detections = merge_nearby_boxes(detections)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        if w > max_width or h > max_height:
            continue
        if area > max_area or area < min_area:
            continue
        if aspect_ratio > 2.5 or aspect_ratio < 0.2:
            continue
        if h < 30 or w < 30:
            continue
        if h > w * 2:
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center = (cx, cy)

        det_class = track.det_class
        if det_class is None:
            continue

        label = det_class
        track_id_to_label[track_id] = label

        violated = False
        if label == 'car':
            if not is_inside(center, car_lane):
                color = colors['violation']
                violated = True
            else:
                color = colors['car']
        elif label == 'motorcycle':
            if not is_inside(center, motorcycle_lane):
                color = colors['violation']
                violated = True
            else:
                color = colors['motorcycle']

        if violated and track_id not in violated_ids:
            violated_ids.add(track_id)
            output_txt.write(f"{track_id},{label}\n")
            output_txt.flush()

            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(width, x2)
            y2_crop = min(height, y2)
            vehicle_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            save_folder = f"violated_vehicles/{label}"
            save_path = f"{save_folder}/{track_id}_{label}.jpg"
            cv2.imwrite(save_path, vehicle_img)
            print(f"🚗 Saved violated vehicle image: {save_path}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, center, 5, color, -1)

    cv2.polylines(frame, [car_lane], True, (255, 100, 100), 2)
    cv2.polylines(frame, [motorcycle_lane], True, (100, 255, 100), 2)

    cv2.imshow("Real-time Violation Detection", frame)

        # Ghi frame đã vẽ vào file video output
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Processed frame {frame_id}")

cap.release()
out.release()

output_txt.close()
cv2.destroyAllWindows()
print("✅ Kết thúc video và lưu danh sách + ảnh xe vi phạm.")
