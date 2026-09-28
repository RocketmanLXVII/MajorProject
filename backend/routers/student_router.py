"""
MulTiCheat Pro — Student & Desk Mapping Router

Endpoints for managing student profiles, seat assignments, and track histories.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

student_router = APIRouter(prefix="/api/students", tags=["Students & Desks"])


class StudentProfile(BaseModel):
    id: str = Field(..., example="STU-001")
    name: str = Field(..., example="John Doe")
    roll_number: str = Field(..., example="CS2026-042")
    assigned_seat: Optional[str] = Field(None, example="Row-2-Desk-3")
    notes: Optional[str] = None


class DeskAssignment(BaseModel):
    desk_id: str = Field(..., example="DESK-005")
    student_id: str = Field(..., example="STU-001")
    polygon: List[List[float]] = Field(..., example=[[100, 200], [200, 200], [200, 300], [100, 300]])


# In-memory storage for local runtime
STUDENT_DB: Dict[str, StudentProfile] = {
    "STU-001": StudentProfile(id="STU-001", name="Alice Smith", roll_number="CS-101", assigned_seat="Row-1-Desk-1"),
    "STU-002": StudentProfile(id="STU-002", name="Bob Jones", roll_number="CS-102", assigned_seat="Row-1-Desk-2"),
    "STU-003": StudentProfile(id="STU-003", name="Charlie Brown", roll_number="CS-103", assigned_seat="Row-2-Desk-1"),
}

DESK_DB: List[DeskAssignment] = []


@student_router.get("/", response_model=List[StudentProfile])
async def list_students():
    """List all registered students in exam database."""
    return list(STUDENT_DB.values())


@student_router.post("/", response_model=StudentProfile, status_code=status.HTTP_201_CREATED)
async def create_student(student: StudentProfile):
    """Register new student profile."""
    if student.id in STUDENT_DB:
        raise HTTPException(status_code=400, detail="Student ID already exists.")
    STUDENT_DB[student.id] = student
    return student


@student_router.get("/{student_id}", response_model=StudentProfile)
async def get_student(student_id: str):
    """Get student profile details."""
    if student_id not in STUDENT_DB:
        raise HTTPException(status_code=404, detail="Student not found.")
    return STUDENT_DB[student_id]


@student_router.post("/desks/assign", status_code=status.HTTP_200_OK)
async def assign_desk(desk: DeskAssignment):
    """Assign student to classroom desk polygon."""
    DESK_DB.append(desk)
    if desk.student_id in STUDENT_DB:
        STUDENT_DB[desk.student_id].assigned_seat = desk.desk_id
    return {"status": "assigned", "desk_id": desk.desk_id, "student_id": desk.student_id}


@student_router.get("/desks/list")
async def list_desks():
    """List all mapped classroom desk polygons."""
    return DESK_DB
