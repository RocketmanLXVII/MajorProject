"""
MulTiCheat Pro — Student & Seat Assignment Pydantic Schemas
"""

from typing import List, Optional
from pydantic import BaseModel


class StudentCreate(BaseModel):
    roll_number: str
    name: str
    face_embedding: Optional[List[float]] = None


class StudentResponse(BaseModel):
    id: str
    roll_number: str
    name: str

    class Config:
        from_attributes = True


class SeatAssignmentRequest(BaseModel):
    student_id: str
    seat_number: int
    desk_bbox: Optional[List[int]] = None
