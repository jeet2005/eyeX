"""
Behavior Detection Service using YOLOv8-pose

Detects student behaviors in classroom:
- studying: Upright posture, looking forward
- sleeping: Head down on desk
- distracted: Looking away, turning around
- focused: Engaged, writing/reading
"""
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

import config


@dataclass
class BehaviorDetection:
    """Represents a detected behavior"""
    person_id: int  # Index of person in frame
    behavior: str   # studying, sleeping, distracted, focused
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    keypoints: Optional[np.ndarray] = None


class BehaviorDetector:
    """
    YOLOv8-pose based behavior detector for classroom monitoring.
    
    Uses body keypoints to classify student behavior:
    - Keypoint indices (COCO format):
      0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
      5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
      9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self):
        self._model = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model_path = config.MODELS_DIR / "yolov8n-pose.pt"
    
    def initialize(self) -> bool:
        """Initialize the YOLOv8-pose model"""
        if self._initialized:
            return True
        
        try:
            from ultralytics import YOLO
            
            # Download model if not exists
            if not self._model_path.exists():
                print("📥 Downloading YOLOv8-nano pose model...")
                self._model = YOLO('yolov8n-pose.pt')
                # Save to models directory
                self._model_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                self._model = YOLO(str(self._model_path))
            
            self._initialized = True
            print("✅ Behavior Detector initialized (YOLOv8-pose)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Behavior Detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[BehaviorDetection]:
        """
        Detect behaviors in frame.
        
        Args:
            image: BGR image
            
        Returns:
            List of BehaviorDetection objects
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        try:
            # Run pose detection with lower confidence for distant/small faces
            results = self._model(image, verbose=False, conf=0.15)
            
            if not results or len(results) == 0:
                return []
            
            detections = []
            result = results[0]
            
            # Check if keypoints are available
            if result.keypoints is None or len(result.keypoints) == 0:
                return []
            
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            
            for i, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                # kpts shape: (17, 3) - x, y, confidence for each keypoint
                behavior, confidence = self._classify_behavior(kpts)
                
                # Convert box from xyxy to xywh
                x1, y1, x2, y2 = box[:4]
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                detections.append(BehaviorDetection(
                    person_id=i,
                    behavior=behavior,
                    confidence=confidence,
                    bbox=bbox,
                    keypoints=kpts
                ))
            
            return detections
            
        except Exception as e:
            print(f"Error in behavior detection: {e}")
            return []
    
    def _classify_behavior(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        Advanced behavior classification using skeletal analysis.
        
        Analyzes: 
        - Head posture (Sleep detection)
        - Gaze direction (Distraction)
        - Body orientation (Focus)
        - Hand gestures (Participation/Studying)
        """
        # Extract key points with confidence
        nose = keypoints[self.NOSE]
        left_eye = keypoints[self.LEFT_EYE]
        right_eye = keypoints[self.RIGHT_EYE]
        left_ear = keypoints[self.LEFT_EAR]
        right_ear = keypoints[self.RIGHT_EAR]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        # Helper to check visibility
        def is_visible(pt): return pt[2] > 0.3
        
        # Visibility flags
        has_face = is_visible(nose) or (is_visible(left_eye) and is_visible(right_eye))
        has_shoulders = is_visible(left_shoulder) and is_visible(right_shoulder)
        has_hands = is_visible(left_wrist) or is_visible(right_wrist)
        
        scores = {
            "studying": 0.0,
            "focused": 0.0,
            "distracted": 0.0,
            "sleeping": 0.0,
            "raising_hand": 0.0
        }
        
        if not has_shoulders:
            # Can't determine much without shoulders
            return "unknown", 0.0

        # Metrics
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 1. SLEEPING Check (Head Level)
        # If nose is below shoulders -> Definitely sleeping/head down
        if has_face and nose[1] > shoulder_mid_y:
            scores["sleeping"] += 0.9
        
        # If head is very low relative to shoulders (chin on chest)
        # Check ear level vs shoulder
        ear_y = (left_ear[1] + right_ear[1]) / 2 if (is_visible(left_ear) and is_visible(right_ear)) else 0
        if ear_y > 0 and (ear_y - shoulder_mid_y) > -shoulder_width * 0.1:
             # Ears are almost at shoulder level
             scores["sleeping"] += 0.4
             
        # 2. RAISING HAND Check
        # Wrist significantly above nose/eyes
        ref_y = nose[1] if is_visible(nose) else shoulder_mid_y
        if has_hands:
            if (is_visible(left_wrist) and left_wrist[1] < ref_y) or \
               (is_visible(right_wrist) and right_wrist[1] < ref_y):
                scores["raising_hand"] += 0.8
                scores["focused"] += 0.4 # Participating is focused

        # 3. DISTRACTED Check (Head Turn)
        # Check horizontal offset of nose from shoulder center
        if has_face:
            nose_offset = abs(nose[0] - shoulder_mid_x)
            offset_ratio = nose_offset / shoulder_width
            
            if offset_ratio > 0.4:
                scores["distracted"] += 0.7 # Looking way to side
            elif offset_ratio > 0.25:
                scores["distracted"] += 0.3
            else:
                scores["focused"] += 0.3 # Centered
        
        # Check Ear Asymmetry (Side profile)
        if is_visible(left_ear) and not is_visible(right_ear):
             scores["distracted"] += 0.4 # Turned right
        elif is_visible(right_ear) and not is_visible(left_ear):
             scores["distracted"] += 0.4 # Turned left

        # 4. STUDYING Check (Head down slightly + Hands visible)
        # Head is looking down but not ON desk (nose above shoulders but verification needed)
        # Eye level lower than usual relative to ears? (Hard with 2D)
        if scores["sleeping"] < 0.5 and scores["distracted"] < 0.5:
            # If hands are active (near desk level?) - we assume desk is below shoulders
            if has_hands:
                # Wrists below shoulders but visible often means writing
                if (is_visible(left_wrist) and left_wrist[1] > shoulder_mid_y) or \
                   (is_visible(right_wrist) and right_wrist[1] > shoulder_mid_y):
                    scores["studying"] += 0.5
            
            # Default to focused if looking forward
            if scores["focused"] > 0.2:
                scores["studying"] += 0.3

        # Select best behavior
        best_behavior = "unknown"
        max_score = 0.0
        
        # Prioritize 'raising_hand' if score is high
        if scores["raising_hand"] > 0.6:
            return "raising_hand", scores["raising_hand"]
            
        for behavior, score in scores.items():
            if score > max_score:
                max_score = score
                best_behavior = behavior
                
        # Fallbacks
        if best_behavior == "unknown" and max_score == 0:
            if has_face:
                best_behavior = "focused" # Innocent until proven guilty
                max_score = 0.5
            else:
                max_score = 0.0
                
        return best_behavior, min(max_score, 1.0)
            

    
    async def detect_async(self, image: np.ndarray) -> List[BehaviorDetection]:
        """Detect behaviors asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.detect, image)
    
    def get_behavior_stats(self, detections: List[BehaviorDetection]) -> Dict[str, int]:
        """Get count of each behavior type"""
        stats = {"studying": 0, "sleeping": 0, "distracted": 0, "focused": 0, "unknown": 0, "total": 0}
        stats["total"] = len(detections)
        for det in detections:
            if det.behavior in stats:
                stats[det.behavior] += 1
        return stats


# Singleton instance
behavior_detector = BehaviorDetector()
