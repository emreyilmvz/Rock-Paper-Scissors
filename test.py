import cv2
import torch
from ultralytics import YOLO

class RockPaperScissorsDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.cap = cv2.VideoCapture(0)
        
    def run(self):
        print("Camera active. Press 'q' to exit.")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Camera connection error!")
                break

            annotated_frame = self.process_frame(frame)
            cv2.imshow("Rock Paper Scissors Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()

    def process_frame(self, frame):
        results = self.model(frame, conf=0.25, verbose=False) 
        
        for result in results:
            frame = result.plot()
            
        return frame

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("System shut down.")

if __name__ == "__main__":
    MODEL_PATH = 'models/colab_best.pt'
    detector = RockPaperScissorsDetector(MODEL_PATH)
    detector.run()