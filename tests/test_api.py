"""
Integration Tests for REST Endpoints (Students, WebSockets, Analysis)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_list_students_endpoint():
    response = client.get("/api/students/")
    assert response.status_code == 200
    students = response.json()
    assert isinstance(students, list)
    assert len(students) >= 3


def test_create_student_endpoint():
    new_student = {
        "id": "STU-TEST-99",
        "name": "Test Student",
        "roll_number": "TEST-99",
        "assigned_seat": "Desk-99",
    }
    response = client.post("/api/students/", json=new_student)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "STU-TEST-99"


def test_assign_desk_endpoint():
    desk_assignment = {
        "desk_id": "DESK-100",
        "student_id": "STU-TEST-99",
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    response = client.post("/api/students/desks/assign", json=desk_assignment)
    assert response.status_code == 200
    assert response.json()["status"] == "assigned"
