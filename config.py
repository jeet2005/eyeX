"""
Configuration settings for Smart Attendance System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STUDENTS_DIR = DATA_DIR / "students"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STUDENTS_DIR.mkdir(exist_ok=True)

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Database (SQLite - legacy, kept for backup)
DATABASE_URL = f"sqlite+aiosqlite:///{DATA_DIR}/attendance.db"

# MongoDB Atlas
MONGODB_URI = "mongodb+srv://jeetsavaliya1908_db_user:9Q2EbGK4HoRLZee4@cluster0.mjae9id.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DATABASE = "eye_x"

# Face Detection (YuNet)
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_DETECTION_THRESHOLD = 0.30
FACE_DETECTION_INPUT_SIZE = (320, 320)

# Face Recognition (InsightFace)
INSIGHTFACE_MODEL = "buffalo_l"
FACE_RECOGNITION_THRESHOLD = 0.42  # Cosine similarity threshold

# Behavior Detection (YOLOv8)
YOLO_MODEL = "yolov8n-pose.pt"
BEHAVIOR_CAPTURE_INTERVAL = 10  # seconds

# WebRTC
ICE_SERVERS = []  # Empty for LAN-only connections
VIDEO_CODEC = "VP8"
AUDIO_CODEC = "opus"

# Audio announcements
ANNOUNCEMENT_LANGUAGE = "en"
ANNOUNCEMENT_RATE = 150  # Words per minute

# Gate camera settings
GATE_DETECTION_INTERVAL = 0.5  # seconds - detect faces every 0.5s
GATE_RECOGNITION_COOLDOWN = 5  # seconds - prevent duplicate recognition

# Classroom camera settings (optional feature)
CLASSROOM_ENABLED = False  # Set to True to enable classroom monitoring
