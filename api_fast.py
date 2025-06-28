from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from model_fast import predict_news_fast
import time

app = FastAPI(title="Fast Fake News Detector")

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: NewsInput, request: Request):
    # Input validation
    if not data.text or not isinstance(data.text, str) or len(data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text is required.")
    
    try:
        # Use the fast model
        label, confidence, explanation = predict_news_fast(data.text)
        return {"label": label, "confidence": confidence, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Fast Fake News Detector API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "fast"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 