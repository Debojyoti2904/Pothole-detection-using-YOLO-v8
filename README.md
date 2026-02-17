# 🕳️ Pothole Detection using YOLOv8 and Streamlit

A complete **Deep Learning-based Pothole Detection System** using **YOLOv8**, **OpenCV**, and **Streamlit** to automatically detect and locate potholes from road images or live camera feeds. 
This project covers data preparation, model training, prediction, and deployment via an interactive web interface.

---

## 🚀 Project Overview

Potholes are one of the primary causes of road damage and traffic accidents. 
This system leverages **YOLOv8 (You Only Look Once)** for real-time object detection to identify potholes efficiently in images, videos, or live streams.

### 🔍 Key Features
- **YOLOv8-based object detection**
- Detect potholes in images, videos, or webcam feed
- **Streamlit interface** for user-friendly interaction
- Customizable configuration using YAML
- Save and analyze detection results

---

## 🧩 Project Structure

```text
pothole-detection/
│
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── pothole_config.yaml    # Model configuration
├── yolov8n.pt            # Pretrained YOLOv8 model weights
│
├── Pothole-Dataset/       # Dataset folder
│
├── src/
│   ├── app.py             # Streamlit app for detection
│   ├── train.py           # Model training script
│   ├── predict.py         # Inference/prediction script
│   ├── split_data.py      # Train-test split helper
│   └── my_converter.py    # Annotation/data converter
│
├── results/               # Saved detection outputs
├── runs/                  # YOLOv8 training logs
├── sample/                # Sample test images
└── venv/                  # Virtual environment (ignored)