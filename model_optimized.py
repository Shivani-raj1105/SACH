"""
model_optimized.py: Optimized version using existing saved model with better performance.
"""
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load the existing saved model (no new downloads needed)
print("Loading optimized model...")
tokenizer = BertTokenizer.from_pretrained("saved_model")
model = BertForSequenceClassification.from_pretrained("saved_model")
model.eval()

# Optimize for inference
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU acceleration")
else:
    print("Using CPU (GPU not available)")

def predict_news_optimized(text):
    """Optimized prediction using existing model."""
    try:
        # Shorter max length for faster processing
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict with optimizations
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            label = "Real" if pred.item() == 1 else "Fake"
        
        # Simple explanation (faster than LIME)
        words = text.split()[:8]  # First 8 words
        explanation = [(word, 0.1) for word in words if len(word) > 2]  # Filter short words
        
        return label, confidence.item(), explanation
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown", 0.0, []

def predict_proba_optimized(texts):
    """Optimized probability prediction."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

if __name__ == "__main__":
    # Test the optimized model
    test_text = "This is a test news article for fake news detection."
    print("Testing optimized model...")
    label, conf, exp = predict_news_optimized(test_text)
    print(f"Result: {label} (confidence: {conf:.2%})")
    print("Optimized model ready!") 