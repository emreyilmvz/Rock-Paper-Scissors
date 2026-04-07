"""
Rock Paper Scissors - Real-Time Inference Script

This script runs real-time object detection using the optimized YOLOv8s model
via the system's webcam. 
"""

import cv2
import torch
from ultralytics import YOLO

def run_inference():
    # Dynamic hardware check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"System running on: {device.upper()}")

    # Load the optimized model
    model = YOLO('models/colab_best.pt').to(device)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    print("Camera is starting, please wait...")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame, exiting...")
            break

        # Run YOLO inference (verbose=False keeps the terminal clean)
        results = model(frame, conf=0.5, verbose=False)

        # Plot the results on the frame
        for result in results:
            annotated_frame = result.plot()

        # Display the frame
        cv2.imshow("Real-time Detection", annotated_frame)

        # Exit condition: Press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("System shut down.")

if __name__ == '__main__':
    run_inference()