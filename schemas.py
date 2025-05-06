from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    pdf_path: Optional[str] = None

class AnswerResponse(BaseModel):
    success: bool
    answer: str
    confidence: float
    error: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    file_path: str
    size: int