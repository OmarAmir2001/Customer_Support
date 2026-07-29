from pydantic import BaseModel

class Student_Profile(BaseModel):
    student_id: str
    name: str
    department: str
    gpa: float
    preferred_language: str