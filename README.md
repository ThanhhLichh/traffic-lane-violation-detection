# 🚦 Lane Violation Detection System

## 🧩 Introduction
This project is an AI-based traffic monitoring system that detects **motorbikes and cars violating lane rules** using **YOLOv8** and **OpenCV**.  
The system processes traffic videos in real time, identifies vehicles, determines their lane positions, and captures evidence when a violation occurs.  
It can be applied in intelligent transportation systems (ITS) and traffic surveillance automation.

---

## ⚙️ Technologies Used
| Technology | Description |
|-------------|-------------|
| **Python 3.x** | Main programming language |
| **OpenCV** | Image processing and lane detection |
| **YOLOv8 (Ultralytics)** | Object detection model for vehicles |
| **NumPy** | Lane zone data storage (.npy) |
| **Matplotlib** | Visualization and debugging support |



## 🚀 How to Use

### 1️⃣ Install dependencies
Run the following command to install required libraries:

`pip install ultralytics opencv-python numpy matplotlib`

### 2️⃣ Select lane areas
Run:
`python lane_selector.py`

Use your mouse to mark valid lane regions for cars and motorbikes.  
The coordinates of each lane will be automatically saved as `.npy` files in the `lanes/` directory:

- `lanes/car_lane.npy` → lane area for cars  
- `lanes/motorcycle_lane.npy` → lane area for motorbikes


### 3️⃣ Run lane violation detection
Run:  
`python violation_detector.py`

This script will:
- Load the **YOLOv8 model** (`yolov8m.pt`)
- Read the input video (e.g., `duong_pho4.mp4`)
- Detect each vehicle’s position and check whether it stays within the correct lane
- Highlight vehicles:
  - ✅ Green box → correct lane  
  - 🚨 Red box → lane violation detected  
- Save violation snapshots automatically to:
  - 📁 `violated_vehicles/car/`
  - 📁 `violated_vehicles/motorcycle/`
  - and log details in 📝 `violations.txt`

---

> “AI-powered traffic monitoring for safer and smarter roads.” 🚗💡

