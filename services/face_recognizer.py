"""
Face Recognition Service using InsightFace

Handles face embedding generation and matching for student recognition.
"""
import asyncio
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import os

import config


@dataclass
class RecognitionMatch:
    """Result of a face recognition match"""
    student_id: Union[int, str]
    name: str
    roll_number: str
    similarity: float


class FaceRecognizer:
    """
    InsightFace-based face recognition service.
    
    Uses ArcFace embeddings for face matching with cosine similarity.
    """
    
    def __init__(self):
        self._model = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._enrolled_faces = {}  # student_id -> (name, roll_number, embedding)
        self._similarity_threshold = 0.35  # Cosine similarity threshold (lower = more lenient)
    
    def initialize(self) -> bool:
        """Initialize the InsightFace recognition model"""
        if self._initialized:
            return True
        
        try:
            from insightface.app import FaceAnalysis
            
            self._model = FaceAnalysis(
                name='buffalo_l',
                root=str(config.MODELS_DIR),
                providers=['CPUExecutionProvider']
            )
            self._model.prepare(ctx_id=0, det_size=(640, 640))
            
            self._initialized = True
            print("Face Recognizer initialized (InsightFace)")
            
            # Load enrolled faces from database
            asyncio.create_task(self._load_enrolled_faces())
            asyncio.create_task(self._load_mongodb_faces())
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Face Recognizer: {e}")
            return False
    
    async def _load_enrolled_faces(self):
        """Load enrolled faces from database"""
        try:
            from database.database import get_session
            from database.models import Student
            from sqlalchemy import select
            
            async with get_session() as session:
                result = await session.execute(select(Student).where(Student.face_embedding.isnot(None)))
                students = result.scalars().all()
                
                for student in students:
                    if student.face_embedding:
                        embedding = np.frombuffer(student.face_embedding, dtype=np.float32)
                        self._enrolled_faces[student.id] = (
                            student.name,
                            student.roll_number,
                            embedding
                        )
                
                print(f"Loaded {len(self._enrolled_faces)} enrolled faces from PostgreSQL")
        except Exception as e:
            print(f"Error loading enrolled faces from PostgreSQL: {e}")
    
    async def _load_mongodb_faces(self):
        """Load faces by directly scanning data/students folder"""
        try:
            from database import mongodb
            
            students_dir = config.STUDENTS_DIR
            print(f"[FACE] Scanning folder: {students_dir}")
            
            if not students_dir.exists():
                print(f"[FACE] ERROR: Folder does not exist: {students_dir}")
                return
            
            # Get all jpg files in the folder
            image_files = list(students_dir.glob("*.jpg"))
            print(f"[FACE] Found {len(image_files)} image files")
            
            if not image_files:
                print("[FACE] No images found in students folder!")
                return
            
            # Get all students from MongoDB for name lookup
            all_students = await mongodb.get_all_students()
            student_map = {str(s["_id"]): s for s in all_students}
            print(f"[FACE] Loaded {len(student_map)} students from MongoDB")
            
            count = 0
            for img_path in image_files:
                # Extract student ID from filename (e.g. "697e7c49d07a20db49e589b9_0.jpg" -> "697e7c49d07a20db49e589b9")
                filename = img_path.stem  # "697e7c49d07a20db49e589b9_0"
                student_id = filename.split("_")[0]  # "697e7c49d07a20db49e589b9"
                
                # Load the image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[FACE] ERROR: Cannot read image: {img_path}")
                    continue
                
                # Generate embedding
                embedding = self.get_embedding(img)
                if embedding is None:
                    print(f"[FACE] ERROR: No face detected in: {img_path}")
                    continue
                
                # Get student info from MongoDB
                student_info = student_map.get(student_id, {})
                name = student_info.get("name", filename)  # Use filename if no name in DB
                roll = student_info.get("roll_number", "")
                
                # Store the embedding (accumulate multiple embeddings per student)
                if student_id not in self._enrolled_faces:
                    self._enrolled_faces[student_id] = (name, roll, [embedding])
                    count += 1
                else:
                    # Add to existing embeddings list
                    existing = self._enrolled_faces[student_id]
                    existing[2].append(embedding)
                
                print(f"[FACE] SUCCESS: Loaded face for '{name}' from {img_path.name}")
            
            # Print summary
            total_embeddings = sum(len(data[2]) for data in self._enrolled_faces.values())
            print(f"[FACE] ========================================")
            print(f"[FACE] TOTAL: {count} students, {total_embeddings} face embeddings")
            print(f"[FACE] ========================================")
            
        except Exception as e:
            import traceback
            print(f"[FACE] ERROR loading faces: {e}")
            traceback.print_exc()
    
    def get_enrolled_count(self) -> int:
        """Get number of enrolled students"""
        return len(self._enrolled_faces)
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image.
        
        Args:
            face_image: BGR image containing a face
            
        Returns:
            512-dimensional embedding or None
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            faces = self._model.get(face_image)
            if faces and len(faces) > 0:
                return faces[0].embedding
            return None
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def recognize(self, face_image: np.ndarray) -> Optional[RecognitionMatch]:
        """
        Recognize a face by comparing to enrolled faces.
        
        Args:
            face_image: BGR image containing a face
            
        Returns:
            RecognitionMatch if found, None otherwise
        """
        # Check if we have any enrolled faces
        if not self._enrolled_faces:
            print("[RECOGNIZE] WARNING: No enrolled faces to compare against!")
            return None
        
        embedding = self.get_embedding(face_image)
        if embedding is None:
            print("[RECOGNIZE] Could not extract embedding from face image")
            return None
        
        best_match = None
        best_similarity = self._similarity_threshold  # 0.35
        best_name = "None"
        
        for student_id, (name, roll_number, embeddings_list) in self._enrolled_faces.items():
            # Compare against ALL embeddings for this student
            for enrolled_embedding in embeddings_list:
                similarity = self._cosine_similarity(embedding, enrolled_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
                    best_match = RecognitionMatch(
                        student_id=student_id,
                        name=name,
                        roll_number=roll_number,
                        similarity=similarity
                    )
        
        if best_match:
            print(f"[RECOGNIZE] MATCH: {best_match.name} ({best_match.similarity:.2f})")
        else:
            print(f"[RECOGNIZE] No match. Best was '{best_name}' with similarity {best_similarity:.2f} (threshold: {self._similarity_threshold})")
        
        return best_match
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def enroll_student(self, student_id: Union[int, str], name: str, roll_number: str, 
                      face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Enroll a student's face.
        
        Args:
            student_id: Database ID
            name: Student name
            roll_number: Roll number
            face_image: BGR image containing the face
            
        Returns:
            (success, message)
        """
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return False, "No face detected in image"
        
        # Store in memory
        self._enrolled_faces[student_id] = (name, roll_number, embedding)
        
        return True, "Face enrolled successfully"
    
    async def recognize_async(self, face_image: np.ndarray) -> Optional[RecognitionMatch]:
        """Async version of recognize"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.recognize, face_image)
    
    def close(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)


# Singleton instance
face_recognizer = FaceRecognizer()
