"""
model_fast.py: A faster version using a smaller model for better performance.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Use a smaller, faster model
MODEL_NAME = "distilbert-base-uncased"  # Much smaller than BERT

# Load tokenizer and model
print("Loading fast model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.eval()

def predict_news_fast(text):
    """Fast prediction using DistilBERT."""
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            label = "Real" if pred.item() == 1 else "Fake"
        
        # Simple explanation (faster than LIME)
        words = text.split()[:10]  # First 10 words
        explanation = [(word, 0.1) for word in words]  # Simple placeholder
        
        return label, confidence.item(), explanation
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown", 0.0, []

def predict_proba_fast(texts):
    """Fast probability prediction."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

if __name__ == "__main__":
    # Test the fast model
    test_text = "This is a test news article for fake news detection."
    print("Testing fast model...")
    label, conf, exp = predict_news_fast(test_text)
    print(f"Result: {label} (confidence: {conf:.2%})")
    print("Fast model ready!") 