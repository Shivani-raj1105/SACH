import streamlit as st
import pandas as pd
import os
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import re

# Set page config
st.set_page_config(
    page_title="Luxe Fake News Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for luxury look
st.markdown("""
<style>
body, .stApp {
    background-color: #181818;
    color: #e5c07b;
    font-family: 'Georgia', serif;
}
.sidebar .sidebar-content {
    background: #222;
    color: #e5c07b;
}
.block-container {
    background: #222;
    border-radius: 18px;
    box-shadow: 0 4px 32px 0 rgba(0,0,0,0.25);
    padding: 2rem 2rem 2rem 2rem;
}
.lux-title {
    font-size: 2.8rem;
    font-family: 'Georgia', serif;
    color: #FFD700;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #000, 0 0px 2px #FFD700;
    margin-bottom: 0.5em;
}
.lux-card {
    background: #181818;
    border: 1.5px solid #FFD700;
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1.2rem 1.5rem;
    box-shadow: 0 2px 16px 0 rgba(255,215,0,0.08);
    margin-bottom: 1.5rem;
}
.lux-label {
    font-size: 2.2rem;
    font-weight: bold;
    color: #FFD700;
    text-shadow: 0 1px 4px #000;
}
.lux-fake {
    color: #ff4c4c;
    font-size: 2.2rem;
    font-weight: bold;
    text-shadow: 0 1px 4px #000;
}
.lux-real {
    color: #50fa7b;
    font-size: 2.2rem;
    font-weight: bold;
    text-shadow: 0 1px 4px #000;
}
.lux-progress .stProgress > div > div > div {
    background-image: linear-gradient(90deg, #FFD700, #e5c07b);
}
.lux-footer {
    color: #FFD700;
    text-align: center;
    font-size: 1.1rem;
    margin-top: 2rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer"""
    try:
        # Check if model files exist
        model_path = "saved_model"
        if not os.path.exists(model_path):
            st.error("Model files not found! Please ensure the saved_model directory exists.")
            return None, None
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set to evaluation mode
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_text(text, model, tokenizer):
    """Make prediction on input text"""
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Map prediction to label
        label = "Fake" if prediction == 1 else "Real"
        
        return label, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0

def get_lime_explanation(text, model, tokenizer, num_features=10):
    """Get LIME explanation for the prediction"""
    try:
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])
        
        def predictor(texts):
            results = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    results.append(probabilities[0].numpy())
            return np.array(results)
        
        exp = explainer.explain_instance(text, predictor, num_features=num_features, num_samples=100)
        
        # Get word importance scores
        word_weights = [(word, weight) for word, weight in exp.as_list()]
        return word_weights
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")
        return []

# Load model on first run
if not st.session_state.model_loaded:
    with st.spinner("Loading AI model... This may take a moment on first run."):
        model, tokenizer = load_model()
        if model is not None and tokenizer is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")

# Sidebar with luxury style
st.sidebar.markdown("""
<div style='color:#FFD700; font-family:Georgia,serif; font-size:1.5rem; text-align:center; margin-bottom:1.5rem;'>
    <b>📰 Luxe News Detector</b>
</div>
<hr style='border:1px solid #FFD700;'>
<div style='color:#e5c07b; font-size:1.1rem;'>
<b>Instructions:</b><br>
- Paste a news headline or article.<br>
- Click <b>Check if it's Fake</b>.<br>
- See the luxurious result below.<br><br>
<b>About:</b><br>
This app uses a BERT-based model to detect fake news with style.<br>
</div>
<hr style='border:1px solid #FFD700;'>
<div style='color:#FFD700; font-size:1rem;'>
<b>Created by:</b> Shivani Raj<br>
<b>Contact:</b> shivani.raj.urs1105@gmail.com</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Detector", "Visualizations"])

if page == "Detector":
    st.markdown('<div class="lux-title">🧠 Luxe Fake News Detector</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("Model is still loading. Please wait...")
        st.stop()
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.subheader("Enter News Text", divider="orange")
        input_text = st.text_area("Paste news headline or article:", height=200)
        analyze = st.button("Check if it's Fake")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result", divider="orange")
        
        if analyze and input_text.strip():
            with st.spinner("Analyzing..."):
                # Make prediction
                label, confidence = predict_text(input_text, st.session_state.model, st.session_state.tokenizer)
                
                # Color-coded, luxury result
                if label == "Fake":
                    st.markdown('<div class="lux-fake">🛑 FAKE</div>', unsafe_allow_html=True)
                elif label == "Real":
                    st.markdown('<div class="lux-real">✅ REAL</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="lux-label">{label}</div>', unsafe_allow_html=True)

                # Confidence progress bar (luxury style)
                st.markdown('<div class="lux-progress">', unsafe_allow_html=True)
                st.progress(min(max(confidence, 0), 1))
                st.markdown('</div>', unsafe_allow_html=True)
                st.write(f"Confidence: <b>{confidence:.2%}</b>", unsafe_allow_html=True)

                # LIME explanation visualization
                if label in ["Fake", "Real"]:
                    st.markdown('<hr style="border:1px solid #FFD700;">', unsafe_allow_html=True)
                    st.markdown('<b>Top Influential Words:</b>', unsafe_allow_html=True)
                    
                    explanation = get_lime_explanation(input_text, st.session_state.model, st.session_state.tokenizer)
                    
                    if explanation:
                        # Create proper dataframe for visualization
                        try:
                            words, weights = zip(*explanation)
                            df_explanation = pd.DataFrame({
                                'Word': words,
                                'Importance': weights
                            })
                            st.bar_chart(df_explanation.set_index('Word'))
                            
                            # Highlight words in the input text
                            highlighted = input_text
                            for word, weight in explanation:
                                if word.strip() and word in highlighted:
                                    color = "#FFD700" if weight > 0 else "#ff4c4c"
                                    highlighted = highlighted.replace(word, f'<span style="background-color:{color};border-radius:4px;">{word}</span>')
                            st.markdown(f'<div style="line-height:1.8;font-size:1.1rem;">{highlighted}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.info("Word analysis available but visualization could not be displayed.")
                            st.write("**Key words:**", ", ".join([word for word, _ in explanation[:5]]))

        elif analyze:
            st.warning("Please enter some text to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class='lux-footer'>
        <hr style='border:1px solid #FFD700;'>
        <small>© 2024 <b>Your Name</b>. All rights reserved. | Luxe Edition</small>
    </div>
    """, unsafe_allow_html=True)

elif page == "Visualizations":
    st.markdown('<div class="lux-title">📊 Visualizations & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.subheader("About the Model", divider="orange")
    
    st.markdown("""
    **Model Information:**
    - **Architecture**: BERT-base-uncased
    - **Task**: Binary classification (Fake/Real)
    - **Input**: Text (max 512 tokens)
    - **Output**: Label + confidence score + explanations
    
    **Features:**
    - Uses LIME for explainability
    - Luxury-themed interface
    - Real-time predictions
    """)
    
    st.markdown('</div>', unsafe_allow_html=True) 