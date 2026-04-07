"""
Rock Paper Scissors - YOLOv8s Training Script

Previous Benchmark (Colab T4 GPU):
- Base Model: yolov8s.pt
- Best Epoch: 69 (Early stopping at 89)
- Overall mAP50: 0.952
- Overall mAP50-95: 0.778
- Paper mAP50: 0.954
- Rock mAP50: 0.952
- Scissors mAP50: 0.951
"""

from ultralytics import YOLO
import torch

def train_model():
    if torch.cuda.is_available():
        print(f"GPU active: {torch.cuda.get_device_name(0)}")
    else:
        print("Error: CUDA not found! Training on CPU will be very slow.")
        return

    # Using the 'Small' model for better accuracy instead of 'Nano'
    model = YOLO('models/yolov8s.pt')

    model.train(
        data='datasets/Rock Paper Scissors SXSW.v14i.yolov8/data.yaml',
        epochs=100,         
        patience=20,        
        imgsz=640,          
        batch=16,           
        project='runs/train',
        name='optimized_model',
        device=0,           
        workers=0           
    )

if __name__ == '__main__':
    train_model()