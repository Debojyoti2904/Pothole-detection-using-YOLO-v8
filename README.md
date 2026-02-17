<div align="center">

  <h1>🚧 Pothole Detection System</h1>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/YOLO-v8-orange.svg" alt="YOLOv8">
    <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/CV-OpenCV-green.svg" alt="OpenCV">
  </p>
  
  <p><i>An advanced Deep Learning solution for real-time road hazard detection.</i></p>

</div>

---

## 🎥 Project Demo
https://github.com/user-attachments/assets/1602fb49-f0b2-4c54-ae64-c26f09dbcff1


---

## 📖 Overview
This project implements an automated **Pothole Detection System** using computer vision and deep learning techniques. By leveraging the **YOLOv8-small** model, we have created a robust and efficient solution for identifying and localizing potholes in road images and videos.

This tool is designed to assist in road maintenance prioritization and driver safety systems.

## ✨ Key Features
* **🧠 YOLOv8-small Model:** Utilizes the compact yet powerful YOLOv8 architecture for high-speed object detection.
* **🎥 Multi-format Input:** Capable of processing both static images and video feeds.
* **⚡ Real-time Detection:** Optimized inference speed suitable for edge devices.
* **💻 User-friendly Interface:** Built with **Streamlit** for easy interaction and result visualization.

## 🛠️ Technology Stack
* **Deep Learning:** YOLOv8 (Ultralytics)
* **Computer Vision:** OpenCV, Supervision
* **Data Processing:** NumPy, Pandas
* **Interface:** Streamlit

---

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Debojyoti2904/Pothole-detection-using-YOLO-v8.git](https://github.com/Debojyoti2904/Pothole-detection-using-YOLO-v8.git)
    cd Pothole-detection-using-YOLO-v8
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 🎯 Usage
1.  Launch the Streamlit interface using the command above.
2.  **Upload** an image or video file containing road footage.
3.  Adjust the **Confidence Threshold** slider to filter weak detections.
4.  View the processed output with bounding boxes drawn around detected potholes.

---
<div align="center">
  <p>Developed with ❤️ for Safer Roads</p>
</div>
