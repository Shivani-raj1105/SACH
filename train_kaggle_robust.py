"""
train_kaggle_robust.py: Robust training script for Kaggle with better error handling
"""
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os
import zipfile
import shutil
import time

def download_with_retry(model_name, max_retries=3):
    """Download model/tokenizer with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Downloading {model_name} (attempt {attempt + 1}/{max_retries})...")
            if "tokenizer" in model_name.lower():
                return BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")
            else:
                return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, cache_dir="./cache")
        except Exception as e:
            print(f"[WARNING] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                raise e

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def save_model_for_download():
    """Save model in a way that can be downloaded without Drive"""
    print("[INFO] Preparing model for download...")
    
    # Create a zip file
    with zipfile.ZipFile('saved_model.zip', 'w') as zipf:
        for root, dirs, files in os.walk('saved_model'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, 'saved_model')
                zipf.write(file_path, arcname)
    
    print("[INFO] Model saved as 'saved_model.zip'")
    print("[INFO] You can download it using: files.download('saved_model.zip')")

def main():
    try:
        print("[INFO] Setting up Kaggle environment...")
        
        # Create cache directory
        os.makedirs("./cache", exist_ok=True)
        
        # Install required packages if not already installed
        try:
            import lime
        except ImportError:
            print("[INFO] Installing required packages...")
            os.system("pip install lime shap")
        
        # Download tokenizer and model with retry
        global tokenizer
        tokenizer = download_with_retry("tokenizer")
        print("[INFO] Tokenizer loaded successfully!")
        
        print("[INFO] Loading LIAR dataset...")
        liar_dataset = load_dataset("liar", split="train[:1000]", trust_remote_code=True)
        liar_dataset = liar_dataset.map(lambda e: {"text": e["statement"], "label": 0 if e["label"] in ["false", "pants-fire"] else 1})
        liar_dataset = liar_dataset.map(preprocess, batched=True)
        liar_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        print("[INFO] LIAR dataset loaded.")

        news_data = []
        if os.path.isfile("news_data.csv"):
            print("[INFO] Loading news_data.csv...")
            df_news = pd.read_csv("news_data.csv")
            for _, row in df_news.iterrows():
                news_data.append({"text": str(row["text"]), "label": 1})
        news_dataset = Dataset.from_pandas(pd.DataFrame(news_data)) if news_data else None
        if news_dataset:
            news_dataset = news_dataset.map(preprocess, batched=True)
            news_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print(f"[INFO] Loaded {len(news_dataset)} news samples.")

        feedback_data = []
        if os.path.isfile("feedback.csv"):
            print("[INFO] Loading feedback.csv...")
            df_feedback = pd.read_csv("feedback.csv")
            for _, row in df_feedback.iterrows():
                label = 1 if str(row["model_prediction"]).strip().lower() == "real" else 0
                feedback_data.append({"text": str(row["text"]), "label": label})
        feedback_dataset = Dataset.from_pandas(pd.DataFrame(feedback_data)) if feedback_data else None
        if feedback_dataset:
            feedback_dataset = feedback_dataset.map(preprocess, batched=True)
            feedback_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print(f"[INFO] Loaded {len(feedback_dataset)} feedback samples.")

        datasets = [liar_dataset]
        if news_dataset:
            datasets.append(news_dataset)
        if feedback_dataset:
            datasets.append(feedback_dataset)
        combined_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else liar_dataset
        print(f"[INFO] Total training samples: {len(combined_dataset)}")

        print("[INFO] Initializing model and training arguments...")
        model = download_with_retry("model")
        print("[INFO] Model loaded successfully!")
        
        args = TrainingArguments(
            "./bert-fake-news",
            per_device_train_batch_size=4,  # Increased for Kaggle GPU
            num_train_epochs=2,
            save_steps=100,
            save_total_limit=3,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100
        )
        trainer = Trainer(model=model, args=args, train_dataset=combined_dataset)
        print("[INFO] Starting training...")
        trainer.train()
        print("[INFO] Training complete. Saving model...")
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")
        print("[INFO] Model and tokenizer saved to 'saved_model'.")
        
        # Save model for download without Drive
        save_model_for_download()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print("To download your model, run this command:")
        print("from google.colab import files")
        print("files.download('saved_model.zip')")
        print("="*50)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[INFO] Try running the download steps manually first")

if __name__ == '__main__':
    main() 