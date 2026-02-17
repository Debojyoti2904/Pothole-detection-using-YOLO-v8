import cv2
from ultralytics import YOLO
import argparse
import os

def run_prediction(model_path, source):
    """
    Loads the trained YOLO model and runs it on a source for live preview.

    Args:
        model_path (str): The path to your trained .pt model file.
        source (str): The path to an image/video file or '0' for webcam.
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    is_webcam = source.isdigit() and source == '0'
    if is_webcam:
        source = 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    print(f"Processing video... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("YOLOv8 Pothole Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing finished.")


if __name__ == "__main__":
    # --- THIS BLOCK IS NOW SIMPLIFIED ---
    # It no longer requires an output path.
    
    model_path = 'runs/detect/pothole_detector_yolov8n/weights/best.pt'
    
    # Define the input video file directly.
    source_video_path = "D:\\Pothole_Detection\\sample\\video\\Screen Recording 2025-08-07 234409.mp4"

    # Call the function with just the model and source paths
    run_prediction(model_path, source_video_path)
