# Cloud Training Guide for Fake News Detection Model

## Quick Start: Google Colab (Recommended)

### Step 1: Access Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook

### Step 2: Upload Your Project
1. Upload the following files to Colab:
   - `train_colab.py` (the Colab-compatible training script)
   - `news_data.csv` (if you have it)
   - `feedback.csv` (if you have it)
   - Any other project files

### Step 3: Run Training
```python
# Install dependencies
!pip install transformers datasets torch fastapi uvicorn streamlit requests lime shap pandas apscheduler

# Run training
!python train_colab.py
```

### Step 4: Download Model
The script will automatically create a zip file and prompt you to download the trained model.

---

## Alternative Platforms

### 1. Kaggle Notebooks
- **URL**: [kaggle.com/notebooks](https://www.kaggle.com/notebooks)
- **Free tier**: 30GB RAM, Tesla P100 GPU, 20GB storage
- **Advantages**: Longer runtime, more stable than Colab
- **Setup**: Upload files and run the same training script

### 2. Paperspace Gradient
- **URL**: [gradient.paperspace.com](https://gradient.paperspace.com/)
- **Free tier**: CPU instances
- **Paid**: $0.59/hour for RTX 4000 GPU
- **Setup**: Create a new project, upload files, run training

### 3. Hugging Face Spaces
- **URL**: [huggingface.co/spaces](https://huggingface.co/spaces)
- **Free tier**: CPU training
- **Advantages**: Integrated with transformers library
- **Limitations**: No GPU in free tier

### 4. Lambda Labs
- **URL**: [lambdalabs.com](https://lambdalabs.com/)
- **Starting at**: $0.60/hour for RTX 3090
- **Advantages**: High-end GPUs, pay-per-use

---

## Resource Requirements

Your training script requires:
- **RAM**: ~8-12GB (fits in free Colab/Kaggle)
- **Storage**: ~5-10GB for model and data
- **GPU**: Tesla T4 or better (free in Colab/Kaggle)
- **Runtime**: ~30-60 minutes for 2 epochs

---

## Tips for Cloud Training

### 1. Save Progress Frequently
```python
# Add to training arguments
save_steps=50,  # Save every 50 steps
save_total_limit=2,  # Keep only 2 checkpoints to save space
```

### 2. Monitor GPU Usage
```python
# Check GPU memory
!nvidia-smi

# Monitor during training
import psutil
print(f"RAM Usage: {psutil.virtual_memory().percent}%")
```

### 3. Handle Disconnections
- Use `save_steps` to save checkpoints frequently
- Resume training from last checkpoint if disconnected
- Consider using Colab Pro for longer sessions

### 4. Optimize for Cloud
```python
# Increase batch size for GPU
per_device_train_batch_size=4,  # Instead of 2

# Use mixed precision for faster training
fp16=True,
```

---

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Reduce `max_length` in tokenizer
- Use gradient accumulation

### Runtime Disconnections
- Save checkpoints frequently
- Use Colab Pro for longer sessions
- Consider Kaggle for more stable runtime

### Slow Training
- Enable GPU acceleration
- Use mixed precision training
- Increase batch size if memory allows

---

## Cost Comparison

| Platform | GPU | Cost | Runtime Limit |
|----------|-----|------|---------------|
| Google Colab (Free) | Tesla T4 | $0 | 12 hours |
| Google Colab Pro | Tesla T4/V100 | $10/month | 24 hours |
| Kaggle (Free) | Tesla P100 | $0 | 9 hours |
| Paperspace | RTX 4000 | $0.59/hour | Unlimited |
| Lambda Labs | RTX 3090 | $0.60/hour | Unlimited |

---

## Next Steps

After training in the cloud:
1. Download the trained model
2. Replace your local `saved_model` directory
3. Test the model locally
4. Deploy your application

The cloud-trained model will work exactly the same as a locally trained one! 