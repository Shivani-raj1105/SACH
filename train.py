from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)


def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def main():
    try:
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
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        args = TrainingArguments(
            "./bert-fake-news",
            per_device_train_batch_size=2,
            num_train_epochs=2,
            save_steps=100, 
            save_total_limit=3  
        )
        trainer = Trainer(model=model, args=args, train_dataset=combined_dataset)
        print("[INFO] Starting training...")
        trainer.train()
        print("[INFO] Training complete. Saving model...")
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")
        print("[INFO] Model and tokenizer saved to 'saved_model'.")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    main()