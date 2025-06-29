from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import time
from datetime import datetime

SCRAPE_SCRIPT = 'scrape_news.py'
TRAIN_SCRIPT = 'train.py'
LOG_FILE = 'retrain_log.txt'
RETRAIN_INTERVAL_DAYS = 7  # Retrain every 7 days

def retrain_job():
    with open(LOG_FILE, 'a') as log:
        log.write(f"\nRetraining started at {datetime.now()}\n")
        log.flush()
        # Run news scraping
        log.write("Running news scraping...\n")
        subprocess.run(['python', SCRAPE_SCRIPT], stdout=log, stderr=log)
        # Run model retraining
        log.write("Running model retraining...\n")
        subprocess.run(['python', TRAIN_SCRIPT], stdout=log, stderr=log)
        log.write(f"Retraining finished at {datetime.now()}\n")
        log.flush()

def main():
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_job, 'interval', days=RETRAIN_INTERVAL_DAYS, next_run_time=datetime.now())
    scheduler.start()
    print(f"Retraining scheduler started. Will retrain every {RETRAIN_INTERVAL_DAYS} days.")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

if __name__ == '__main__':
    main() 