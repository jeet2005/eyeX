"""
Face Detection Service using YuNet (OpenCV DNN)
Fast, lightweight, CPU-optimized face detection
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import YUNET_MODEL_PATH, FACE_DETECTION_THRESHOLD, FACE_DETECTION_INPUT_SIZE


@dataclass
class FaceDetection:
    """Represents a detected face"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    
    @property
    def x(self) -> int:
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    def get_center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def get_area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """
    YuNet-based face detector using OpenCV DNN.
    Optimized for CPU with sub-5ms detection on 320x320 images.
    """
    
    def __init__(self, model_path: Optional[Path] = None, threshold: float = None):
        self.model_path = model_path or YUNET_MODEL_PATH
        self.threshold = threshold or FACE_DETECTION_THRESHOLD
        self.input_size = FACE_DETECTION_INPUT_SIZE
        self.detector = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the YuNet face detector"""
        if self._initialized:
            return True
            
        try:
            if not self.model_path.exists():
                print(f"⚠️ YuNet model not found at {self.model_path}")
                print("Please download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet")
                return False
            
            self.detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                self.input_size,
                self.threshold,
                0.3,  # NMS threshold
                5000  # Top K
            )
            
            self._initialized = True
            print(f"✅ YuNet face detector initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize face detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image synchronously.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of FaceDetection objects
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        if image is None or image.size == 0:
            return []
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Update input size if different
        if (w, h) != self.input_size:
            self.detector.setInputSize((w, h))
        
        # Detect faces
        _, faces = self.detector.detect(image)
        
        if faces is None:
            return []
        
        detections = []
        for face in faces:
            # face format: [x, y, w, h, score, landmarks(10 values)]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[-1])
            
            # Extract landmarks if available (5 points, 10 values)
            landmarks = None
            if len(face) >= 14:
                landmarks = face[4:14].reshape(5, 2)
            
            detections.append(FaceDetection(
                bbox=(x, y, fw, fh),
                confidence=confidence,
                landmarks=landmarks
            ))
        
        return detections
    
    async def detect_async(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces asynchronously (non-blocking).
        Runs detection in thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.detect, image)
    
    def draw_detections(self, image: np.ndarray, detections: List[FaceDetection], 
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw detection boxes and landmarks on image"""
        result = image.copy()
        
        for det in detections:
            # Draw bounding box
            cv2.rectangle(
                result,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color,
                2
            )
            
            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(
                result, label,
                (det.x, det.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw landmarks
            if det.landmarks is not None:
                for point in det.landmarks:
                    cv2.circle(result, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        
        return result
    
    def extract_face(self, image: np.ndarray, detection: FaceDetection, 
                     padding: float = 0.5) -> Optional[np.ndarray]:
        """
        Extract and align face region from image.
        
        Args:
            image: Source image
            detection: FaceDetection object
            padding: Relative padding around face (0.2 = 20%)
            
        Returns:
            Cropped and aligned face image
        """
        h, w = image.shape[:2]
        
        # Calculate padded bounding box
        pad_w = int(detection.width * padding)
        pad_h = int(detection.height * padding)
        
        x1 = max(0, detection.x - pad_w)
        y1 = max(0, detection.y - pad_h)
        x2 = min(w, detection.x + detection.width + pad_w)
        y2 = min(h, detection.y + detection.height + pad_h)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        return face
    
    def close(self):
        """Clean up resources"""
        self._executor.shutdown(wait=False)
        self._initialized = False


# Global face detector instance
face_detector = FaceDetector()
