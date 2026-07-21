import os
import shutil
from src.predict import run_edge_inference

from fastapi import FastAPI, File, UploadFile  # Added File and UploadFile here
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (allows Vercel frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Wildlife Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        prediction, confidence = run_edge_inference(temp_file_path)
        
        return {
            "prediction": str(prediction),
            "confidence": round(float(confidence), 4) if confidence else None
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)