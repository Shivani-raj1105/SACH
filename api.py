from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from model import predict_news
import time
import threading

app = FastAPI()

class NewsInput(BaseModel):
    text: str

# Simple in-memory rate limiter
rate_limit = {}
MAX_REQUESTS = 5
WINDOW = 60  # seconds

# Model reload logic
global_model = None
global_tokenizer = None
def reload_model():
    global global_model, global_tokenizer
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        global_tokenizer = BertTokenizer.from_pretrained('saved_model')
        global_model = BertForSequenceClassification.from_pretrained('saved_model')
        print('[INFO] Model and tokenizer reloaded from disk.')
        return True, 'Model reloaded successfully.'
    except Exception as e:
        print(f'[ERROR] Failed to reload model: {e}')
        return False, str(e)

# Initial load
reload_model()

@app.post("/predict")
def predict(data: NewsInput, request: Request):
    # Rate limiting
    ip = request.client.host
    now = time.time()
    if ip not in rate_limit:
        rate_limit[ip] = []
    # Remove timestamps older than WINDOW
    rate_limit[ip] = [t for t in rate_limit[ip] if now - t < WINDOW]
    if len(rate_limit[ip]) >= MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    rate_limit[ip].append(now)
    # Input validation
    if not data.text or not isinstance(data.text, str) or len(data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text is required.")
    try:
        # Use the global model/tokenizer
        from model import predict_news as orig_predict_news
        def predict_news_with_reload(text):
            # Patch model/tokenizer in model.py
            import model
            model.model = global_model
            model.tokenizer = global_tokenizer
            return orig_predict_news(text)
        label, confidence, explanation = predict_news_with_reload(data.text)
        return {"label": label, "confidence": confidence, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/reload-model")
def reload_model_endpoint():
    success, msg = reload_model()
    if success:
        return {"status": "success", "message": msg}
    else:
        raise HTTPException(status_code=500, detail=f"Reload failed: {msg}")