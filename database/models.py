"""
Database models for the Smart Attendance System.
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, LargeBinary, ForeignKey, Text
from sqlalchemy.orm import relationship

from database.database import Base


class Student(Base):
    """Student model for storing student information and face data."""
    
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    roll_number = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    face_embedding = Column(LargeBinary, nullable=True)  # Stored as numpy array bytes
    face_image_path = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    attendances = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Student(id={self.id}, name='{self.name}', roll='{self.roll_number}')>"


class Attendance(Base):
    """Attendance model for tracking student attendance."""
    
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, default=date.today, index=True)
    check_in_time = Column(DateTime, default=datetime.utcnow)
    check_out_time = Column(DateTime, nullable=True)
    status = Column(String(20), default="present")  # present, absent, late, excused
    detection_confidence = Column(Float, nullable=True)
    auto_detected = Column(Boolean, default=True)  # True if detected by AI, False if manual
    notes = Column(Text, nullable=True)
    
    # Relationship
    student = relationship("Student", back_populates="attendances")
    
    def __repr__(self):
        return f"<Attendance(student_id={self.student_id}, date={self.date}, status='{self.status}')>"
