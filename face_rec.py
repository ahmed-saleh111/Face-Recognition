"""
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .

 
"""

import cv2
from deepface import DeepFace
import os
import re

# Parameters for DeepFace
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", #TOP
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet"
]

metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', #TOP but SLOW
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]

alignment_modes = [True, False]

def get_name_from_path(path):
    name = os.path.basename(path).split('.')[0]
    return re.sub(r'\d+$', '', name)  # Remove trailing numbers

def process_face_recognition(frame):
    """Processes a frame for face recognition and annotates results."""
    try:
        dfs = DeepFace.find(
            img_path=frame,
            db_path="face_database_video4", 
            model_name=models[2],
            distance_metric=metrics[0],
            detector_backend=backends[8],
            align=alignment_modes[0],
        )

        print(dfs)
        
        for df in dfs:
            if not df.empty:
                for _, row in df.iterrows():
                    x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
                    name = get_name_from_path(row['identity'])
                    
                    # Draw rectangle
                    cv2.putText(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
                    # Put name
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    except Exception as e:
        print("Error:", e)
    
    return frame


def main():
    cap = cv2.VideoCapture(r"videos\video4.mp4")  # Open the default webcam (0 means primary camera)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 1500, 1200)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process face recognition
        processed_frame = process_face_recognition(frame)

        # Display the processed frame
        cv2.imshow("Face Recognition", processed_frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
