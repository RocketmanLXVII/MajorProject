"""
MulTiCheat Pro — Exam & ExamStudent Junction ORM Models
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import relationship
from backend.models.base import Base


class Exam(Base):
    __tablename__ = "exams"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    room_layout = Column(JSON, nullable=True)  # Calibration, homography, desk polygons
    status = Column(String(20), default="scheduled")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    seat_assignments = relationship("ExamStudent", back_populates="exam", cascade="all, delete-orphan")


class ExamStudent(Base):
    __tablename__ = "exam_students"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exam_id = Column(String(36), ForeignKey("exams.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    seat_number = Column(Integer, nullable=False)
    desk_bbox = Column(JSON, nullable=True)  # [x, y, w, h] in video frame

    exam = relationship("Exam", back_populates="seat_assignments")

    __table_args__ = (
        UniqueConstraint("exam_id", "student_id", name="uq_exam_student"),
        UniqueConstraint("exam_id", "seat_number", name="uq_exam_seat"),
    )
