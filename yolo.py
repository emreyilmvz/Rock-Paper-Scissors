import cv2
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

model = YOLO('models/yolov8n.pt').to('cuda')

kamera = cv2.VideoCapture(0)

print("bekleyin kamera açılıyor")

while True:

    ret, frame = kamera.read()  # Kameradan kare alınıyor
    if not ret:
        print("Kare alınamadı, çıkılıyor...")
        break

    sonuclar = model(frame, conf=0.5)

    for sonuc in sonuclar:
        islenmis_kare = sonuc.plot()

    cv2.imshow("realtime frame", islenmis_kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()

