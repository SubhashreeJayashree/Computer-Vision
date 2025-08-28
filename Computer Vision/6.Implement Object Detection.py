from ultralytics import YOLO
import cv2
model=YOLO("yolov8n.pt") #yolov8s.pt for better accuracy
# Open webcam
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    # Run YOLOv8 tracking
    results=model.track(frame,persist=True,tracker="bytetracker.yaml")
    # Annotate frame with results
    annotated_frame=results[0].plot()
    # Show video in real-time
    cv2.imshow("YOLO Object Tracking",annotated_frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()