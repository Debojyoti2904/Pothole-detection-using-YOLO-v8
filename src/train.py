from ultralytics import YOLO
import torch

def start_training():
    model = YOLO('yolov8n.pt')
    data_config_path = 'pothole_config.yaml'
    training_epochs = 50
    image_size = 640
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'device'}")
    
    print("Starting model training...")
    model.train(
        data=data_config_path,
        epochs=training_epochs,
        imgsz=image_size,
        device=device,
        name='pothole_detector_yolov8n'
    )
    print(f'Training finished')
    
if __name__ == "__main__":
    start_training()