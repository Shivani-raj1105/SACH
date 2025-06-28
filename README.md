# 🧠 Fake News Detector

A sophisticated BERT-based fake news detection system with a beautiful Streamlit web interface and FastAPI backend.

## ✨ Features

- **BERT-based Classification**: Uses fine-tuned BERT model for accurate fake news detection
- **Beautiful UI**: Luxury-themed Streamlit interface with gold accents
- **Explainability**: LIME-based explanations showing influential words
- **Real-time API**: FastAPI backend with rate limiting
- **User Feedback**: Collect user feedback for model improvement
- **Visualizations**: Dashboard for analyzing flagged predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- At least 4GB RAM (for BERT model loading)
- Internet connection (for first-time package installation)

### Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files are present:**
   ```
   saved_model/
   ├── config.json
   ├── model.safetensors
   ├── vocab.txt
   ├── tokenizer_config.json
   └── special_tokens_map.json
   ```

### Running the Application

#### Option 1: Using the Startup Script (Recommended)
```bash
python run.py
```

This will:
- Check all dependencies
- Verify model files
- Start the FastAPI server
- Provide instructions for starting Streamlit

#### Option 2: Manual Startup

1. **Start the API server:**
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **In a new terminal, start the web interface:**
   ```bash
   streamlit run app.py
   ```

3. **Access the application:**
   - Web Interface: http://localhost:8501
   - API Documentation: http://127.0.0.1:8000/docs

## 📖 Usage

### Web Interface
1. Open http://localhost:8501 in your browser
2. Paste a news headline or article text
3. Click "Check if it's Fake"
4. View the prediction with confidence score and explanations
5. Optionally flag incorrect predictions for feedback

### API Usage
```python
import requests

# Make a prediction
response = requests.post("http://127.0.0.1:8000/predict", 
                        json={"text": "Your news text here"})
result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔧 Configuration

### Model Reloading
The API includes an endpoint to reload the model without restarting:
```bash
curl -X POST "http://127.0.0.1:8000/reload-model"
```

### Rate Limiting
- Default: 5 requests per minute per IP
- Configurable in `api.py`

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU issues:**
   The model will automatically use CPU if CUDA is not available.

3. **Port already in use:**
   - Change the port in the startup command
   - Or kill the process using the port

4. **Model loading errors:**
   - Ensure all files in `saved_model/` are present
   - Check file permissions

### Performance Tips

- First prediction may be slow (model loading)
- Subsequent predictions are much faster
- Consider using a GPU for better performance

## 📊 Model Information

- **Architecture**: BERT-base-uncased
- **Task**: Binary classification (Fake/Real)
- **Input**: Text (max 512 tokens)
- **Output**: Label + confidence score + explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Streamlit for the web interface
- LIME for explainability
- BERT model architecture

---

**Note**: This is a demonstration project. For production use, consider additional security measures, model validation, and deployment optimizations. 