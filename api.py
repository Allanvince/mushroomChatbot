from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from test import MushroomChatbot
from schemas import QuestionRequest, AnswerResponse, UploadResponse
import os

app = FastAPI(
    title="Mushroom Farming Q&A API",
    description="API for answering questions from PDF documents",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = MushroomChatbot()

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Main Q&A endpoint"""
    try:
        answer, confidence = chatbot.answer_question(
            question=request.question,
            context=request.context,
            pdf_path=request.pdf_path
        )
        return {
            "success": True,
            "answer": answer,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "confidence": 0.0,
            "error": str(e)
        }

@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """PDF upload endpoint"""
    try:
        # Create uploads directory if not exists
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            
        return {
            "success": True,
            "file_path": file_path,
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)