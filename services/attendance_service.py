"""
Attendance Service - Handles all attendance-related operations.
"""

import logging
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import joinedload

from database.models import Student, Attendance

logger = logging.getLogger(__name__)


class AttendanceService:
    """Service class for managing attendance records."""
    
    def __init__(self):
        self._last_marked = {}  # Track last marked time per student to avoid duplicates
        self._cooldown_seconds = 60  # Minimum seconds between attendance marks
    
    async def mark_present(
        self, 
        session: AsyncSession, 
        student_id: int, 
        confidence: float = 1.0,
        auto_detected: bool = True
    ) -> Tuple[bool, str, Optional[Attendance]]:
        """
        Mark a student as present.
        
        Returns:
            Tuple of (success, message, attendance_record)
        """
        try:
            # Check cooldown to avoid duplicate marks
            now = datetime.now()
            last_time = self._last_marked.get(student_id)
            if last_time and (now - last_time).total_seconds() < self._cooldown_seconds:
                return False, "Already marked recently", None
            
            # Check if already marked today
            today = date.today()
            existing = await session.execute(
                select(Attendance).where(
                    and_(
                        Attendance.student_id == student_id,
                        Attendance.date == today
                    )
                )
            )
            
            if existing.scalar_one_or_none():
                return False, "Already marked present today", None
            
            # Get student
            student_result = await session.execute(
                select(Student).where(Student.id == student_id)
            )
            student = student_result.scalar_one_or_none()
            
            if not student:
                return False, "Student not found", None
            
            # Create attendance record
            attendance = Attendance(
                student_id=student_id,
                date=today,
                check_in_time=now,
                status="present",
                detection_confidence=confidence,
                auto_detected=auto_detected
            )
            
            session.add(attendance)
            await session.commit()
            await session.refresh(attendance)
            
            # Update cooldown tracker
            self._last_marked[student_id] = now
            
            logger.info(f"Marked {student.name} as present (confidence: {confidence:.2f})")
            
            return True, f"{student.name} marked present", attendance
            
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            await session.rollback()
            return False, str(e), None
    
    async def is_marked_today(self, session: AsyncSession, student_id: int) -> bool:
        """Check if student is already marked today."""
        today = date.today()
        
        result = await session.execute(
            select(Attendance).where(
                and_(
                    Attendance.student_id == student_id,
                    Attendance.date == today
                )
            )
        )
        
        return result.scalar_one_or_none() is not None
    
    async def get_today_count(self, session: AsyncSession) -> dict:
        """Get today's attendance count."""
        today = date.today()
        
        # Get total students
        total_result = await session.execute(select(Student).where(Student.is_active == True))
        total_students = len(total_result.scalars().all())
        
        # Get present count
        present_result = await session.execute(
            select(Attendance).where(
                and_(
                    Attendance.date == today,
                    Attendance.status == 'present'
                )
            )
        )
        present_count = len(present_result.scalars().all())
        
        return {
            'total': total_students,
            'present': present_count,
            'absent': total_students - present_count,
            'rate': round((present_count / total_students * 100) if total_students > 0 else 0, 1)
        }
    
    async def get_today_attendance(self, session: AsyncSession) -> list:
        """Get today's attendance records with student details."""
        today = date.today()
        
        # Query attendance with student join
        result = await session.execute(
            select(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .where(Attendance.date == today)
            .order_by(Attendance.check_in_time.desc())
        )
        
        records = []
        for attendance, student in result.all():
            records.append({
                'id': attendance.id,
                'student_id': student.id,
                'name': student.name,
                'roll_number': student.roll_number,
                'status': attendance.status,
                'check_in_time': attendance.check_in_time.strftime('%H:%M:%S') if attendance.check_in_time else None,
                'auto_detected': attendance.auto_detected,
                'confidence': round(attendance.detection_confidence * 100, 1) if attendance.detection_confidence else 0
            })
        
        return records
    
    async def get_attendance_stats(self, session: AsyncSession) -> dict:
        """Get attendance statistics."""
        today = date.today()
        
        # Get total students
        total_result = await session.execute(select(Student).where(Student.is_active == True))
        all_students = total_result.scalars().all()
        total_students = len(all_students)
        
        # Get present count
        present_result = await session.execute(
            select(Attendance).where(
                and_(
                    Attendance.date == today,
                    Attendance.status == 'present'
                )
            )
        )
        present_count = len(present_result.scalars().all())
        
        return {
            'total_students': total_students,
            'present_today': present_count,
            'absent_today': total_students - present_count,
            'attendance_rate': round((present_count / total_students * 100) if total_students > 0 else 0, 1)
        }
    
    async def update_attendance(
        self, 
        session: AsyncSession, 
        attendance_id: int, 
        status: str, 
        notes: str = None
    ) -> tuple:
        """Update an attendance record."""
        result = await session.execute(
            select(Attendance).where(Attendance.id == attendance_id)
        )
        attendance = result.scalar_one_or_none()
        
        if not attendance:
            return False, "Attendance record not found"
        
        attendance.status = status
        if notes:
            attendance.notes = notes
        
        await session.commit()
        return True, "Attendance updated successfully"
    
    async def delete_attendance(self, session: AsyncSession, attendance_id: int) -> tuple:
        """Delete an attendance record."""
        result = await session.execute(
            select(Attendance).where(Attendance.id == attendance_id)
        )
        attendance = result.scalar_one_or_none()
        
        if not attendance:
            return False, "Attendance record not found"
        
        await session.delete(attendance)
        await session.commit()
        return True, "Attendance deleted successfully"


# Singleton instance
attendance_service = AttendanceService()
