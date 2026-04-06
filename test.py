import cv2
import torch
from ultralytics import YOLO

class TasKagitMakasYapayZeka:
    def __init__(self, model_yolu):
        # Donanım kontrolü
        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model yükleme
        self.model = YOLO(model_yolu).to(self.cihaz)
        
        # Kamera başlatma
        self.kamera = cv2.VideoCapture(0)
        
    def calistir(self):
        print("Kamera aktif. Çıkmak için 'q' tuşuna basınız.")
        
        while True:
            basarili_mi, frame = self.kamera.read()
            if not basarili_mi:
                print("Kamera bağlantısı hatası!")
                break

            # Tahminleme ve görselleştirme
            islenmis_kare = self.yapay_zekayi_uygula(frame)
            
            cv2.imshow("Tas Kagit Makas Tespit", islenmis_kare)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.temizle()

    def yapay_zekayi_uygula(self, frame):
        # Tahmin parametreleri: Güven eşiği 0.25, loglar kapalı
        sonuclar = self.model(frame, conf=0.25, verbose=False) 
        
        for sonuc in sonuclar:
            frame = sonuc.plot()
            
        return frame

    def temizle(self):
        self.kamera.release()
        cv2.destroyAllWindows()
        print("Sistem kapatıldı.")

if __name__ == "__main__":
    MODEL_ADRESI = 'runs/detect/train3/weights/best.pt'

    uygulama = TasKagitMakasYapayZeka(MODEL_ADRESI)
    uygulama.calistir()