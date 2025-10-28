# 👨‍👩‍👧 Simple Gender Detection Model

A lightweight **gender detection model** based on **YOLOv11**, designed to predict gender from face images. This is an initial prototype for learning and experimentation and will be updated with improved versions over time.

---

## 🔗 Model

- **Architecture:** [YOLOv11](https://github.com/ultralytics/ultralytics)  
- **Purpose:** Detect gender from face images  
- **Classes:** 
  - `1` - Man 👨  
  - `0` - Woman 👩  

---

## 📦 Dataset

- **Source:** [Face-Gender-Recognition](https://universe.roboflow.com/facegenderdetection/face-gender-recognition-cegug) on Roboflow  
- **Number of Images:**  
  - Training: 823 🖼️  
  - Validation: 223 🖼️  

> **Note:** Dataset is relatively small; the model may not always be accurate.  

---

## ⚡ Features

- Simple implementation, easy to experiment with  
- Lightweight, suitable for small datasets  
- Quick training and inference  
- First prototype for learning purposes  

---

## 🛠️ Installation

1. Clone YOLOv11 repository:
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip install -r requirements.txt
   ```
2. Prepare your dataset in YOLO format.

---

## 🚀 Training
  Train the model on your dataset:
  ```bash
  python train.py
  ```
Predicted bounding boxes will show gender labels:
👨 Man
👩 Woman

---

## ⚠️ Disclaimer
This model is a first prototype, trained on a small dataset using YouTube tutorials. Accuracy is not guaranteed, and predictions may sometimes be incorrect.
---

## 📌 Next Steps
Expand the dataset for better generalization
Optimize YOLOv11 hyperparameters
Explore transfer learning to improve accuracy
Add real-time webcam inference support
Create a web demo or mobile integration

---

## 🌟 Credits

YOLOv11: ultralytics/ultralytics

Dataset: Roboflow – Face-Gender-Recognition

---

## 📸 Demo

Here’s a conceptual example of detection results

## 📸 Demo

Here’s a conceptual example of detection results:

| Input Image      | Output Prediction      |
|-----------------|---------------------|
| input_face.jpg   | output_face.jpg      |
