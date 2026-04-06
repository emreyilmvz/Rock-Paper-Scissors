from ultralytics import YOLO
import torch

def train_model():
    if torch.cuda.is_available():
        print(f"Ekran kartı devrede: {torch.cuda.get_device_name(0)}")
    else:
        print("Hata: CUDA bulunamadı! İşlemci (CPU) ile eğitim çok yavaş olur.")
        return

    model = YOLO('models/yolov8n.pt')

    model.train(
        data='datasets/Rock Paper Scissors SXSW.v14i.yolov8/data.yaml',
        epochs=50,
        imgsz=640,          
        batch=16,           
        device=0,           
        workers=0           
    )

if __name__ == '__main__':
    train_model()