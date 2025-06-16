# School Video Analysis

A professional video analysis tool for face recognition in school environments, built with Streamlit, OpenCV, and DeepFace. This application enables users to upload classroom videos and automatically process them to detect  and recogize faces frame-by-frame using a DeepFace Models.

## Features

- **Video Upload:** Supports MP4 and AVI formats for easyrecogize classroom video uploads.
- **Face Recognition:** Identifies individuals in videos using the DeepFace library with multiple face recognition models.
- **Real-Time Visualization:** Displays processed frames in real-time during analysis.
- **Processed Video Output:** Saves and allows playback of the processed video with detected and recognized faces overlayed.

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- OpenCV
- Ultralytics YOLO (for YOLOv11 models)
- DeepFace (for face recognition)
- Other dependencies as required by your models (see below)

### Installation

It is recommended to use a dedicated conda environment for this project. Below are the steps to set up the environment with Python 3.12:

```bash
conda create -n face_recog-env python=3.12 -y
conda activate face_recog-env
```

Clone this repository:

```bash
git clone <your-repo-url>
cd face_recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Clone and install the DeepFace repository for face recognition:

```bash
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
cd ..
```

Place your trained models in the `models/` directory.

## Usage

- **Start the Streamlit app:**
  ```bash
  streamlit run streamlit.py
  ```
- Upload a classroom video (MP4 or AVI).
- Click **Start processing** to analyze the video for face recognition.
- View real-time results and download the processed video with face annotations.

## Project Structure

```
.
├── streamlit.py             # Streamlit web application
├── face_reg.py              # Face recognition logic using DeepFace
├── models/                  # Pre-trained model files (including YOLOv11 from Ultralytics)
├── videos/                  # Sample videos
├── face_database_video4/    # Database of face images for recognition
└── ...
```
![Emotion Detection](face.gif)

## Using DeepFace for Face Recognition

This project uses the DeepFace library for face recognition.
Supported face recognition models include VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, and GhostFaceNet.
Supported detector backends include opencv, ssd, dlib, mtcnn, fastmtcnn, retinaface, mediapipe, yolov8, yolov11s, yolov11n, yolov11m, yunet, and centerface.

To set up the face database:
- Place reference face images in the `face_database_video4/` directory.
- Ensure filenames are meaningful (e.g., `person_name.jpg`) as they are used to label recognized faces.

Example usage in code:

```python
from deepface import DeepFace
result = DeepFace.find(
    img_path=frame,
    db_path="face_database_video4",
    model_name="Facenet512",
    detector_backend="yolov11s",
    distance_metric="cosine",
    align=True
)
```

## License

This project is intended for educational and research purposes. Please review and comply with your institution's data privacy and ethical guidelines when using classroom videos.
