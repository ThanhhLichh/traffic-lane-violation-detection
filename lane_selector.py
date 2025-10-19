import cv2
import numpy as np
import os

# Hàm vẽ đa giác vùng làn
def draw_polygon(event, x, y, flags, param):
    global drawing, current_lane
    if event == cv2.EVENT_LBUTTONDOWN:
        current_lane.append((x, y))

# Cài đặt
video_path = 'duong_pho3.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

cv2.namedWindow("Draw Lane Regions")
cv2.setMouseCallback("Draw Lane Regions", draw_polygon)

lanes = []
labels = ['car_lane', 'motorcycle_lane']
colors = [(255, 0, 0), (0, 255, 0)]

for i in range(2):
    current_lane = []
    while True:
        temp = frame.copy()
        for pt in current_lane:
            cv2.circle(temp, pt, 5, colors[i], -1)
        if len(current_lane) > 1:
            cv2.polylines(temp, [np.array(current_lane)], False, colors[i], 2)
        cv2.putText(temp, f"Draw {labels[i]} - Press 'Enter' to confirm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Draw Lane Regions", temp)
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            lanes.append(np.array(current_lane))
            break
    print(f"{labels[i]} defined.")

cv2.destroyAllWindows()

# Lưu vùng làn
os.makedirs('lanes', exist_ok=True)
np.save('lanes/car_lane.npy', lanes[0])
np.save('lanes/motorcycle_lane.npy', lanes[1])
print("Saved lane regions.")
