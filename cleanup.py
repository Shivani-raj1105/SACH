import os
import shutil

def ask_and_delete(path, is_dir=False):
    if os.path.exists(path):
        resp = input(f"Delete {path}? (y/n): ").strip().lower()
        if resp == 'y':
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"Deleted {path}")
        else:
            print(f"Kept {path}")

if __name__ == '__main__':
    # Delete old checkpoints
    checkpoints_dir = './bert-fake-news'
    if os.path.isdir(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.startswith('checkpoint-'):
                ask_and_delete(os.path.join(checkpoints_dir, name), is_dir=True)

    ask_and_delete('feedback.csv')

    ask_and_delete('news_data.csv')
    # Delete retrain_log.txt
    ask_and_delete('retrain_log.txt') 
