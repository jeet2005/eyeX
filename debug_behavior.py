import cv2
import sys
import os
import asyncio
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR
from services.behavior_detector import BehaviorDetector

async def test_detection():
    print("----- Behavior Detection Debug Info -----")
    
    # Path to the uploaded image causing issues
    image_path = "C:/Users/jeets/.gemini/antigravity/brain/9c4066fd-1eb0-4d37-b059-18aa09cbe32d/uploaded_media_0_1769887815725.png"
    
    if not os.path.exists(image_path):
        # Fallback to jpg if png not found
        image_path = image_path.replace(".png", ".jpg")
        
    if not os.path.exists(image_path):
        print(f"❌ Image not found at: {image_path}")
        return

    print(f"📸 Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("❌ Failed to load image (cv2.imread returned None)")
        return
        
    print(f"ℹ️ Image Shape: {frame.shape}")
    
    # Initialize detector
    detector = BehaviorDetector()
    if not detector.initialize():
        print("❌ Failed to initialize detector")
        return
        
    print("🧠 Running detection...")
    # Run sync detection for debug
    detections = detector.detect(frame)
    
    print(f"\n📊 Results: {len(detections)} detections")
    
    for i, det in enumerate(detections):
        print(f"  [{i}] Behavior: {det.behavior}, Conf: {det.confidence:.2f}, Box: {det.bbox}")
        
    if len(detections) == 0:
        print("\n⚠️ No detections found! Suggestions:")
        print("1. Lighting might be too poor")
        print("2. Confidence threshold (0.20) might be too high")
        print("3. Person might be too close/far or truncated")
        
if __name__ == "__main__":
    asyncio.run(test_detection())
