import requests
import csv
from datetime import datetime

# Replace with your NewsAPI.org API key
API_KEY = 'aca857d5bd674481a60d3a42657fc136'
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

PARAMS = {
    'language': 'en',
    'pageSize': 100,  # max per request
    'apiKey': API_KEY
}

OUTPUT_FILE = 'news_data.csv'

def fetch_news():
    response = requests.get(NEWS_API_URL, params=PARAMS)
    response.raise_for_status()
    data = response.json()
    articles = data.get('articles', [])
    return articles

def save_to_csv(articles, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'source', 'publishedAt', 'url'])
        for article in articles:
            text = article.get('title', '') + '\n' + (article.get('description') or '')
            source = article.get('source', {}).get('name', '')
            publishedAt = article.get('publishedAt', '')
            url = article.get('url', '')
            writer.writerow([text, source, publishedAt, url])

def main():
    try:
        print(f"[INFO] Fetching news at {datetime.now()}...")
        articles = fetch_news()
        print(f"[INFO] Fetched {len(articles)} articles.")
        save_to_csv(articles, OUTPUT_FILE)
        print(f"[INFO] Saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == '__main__':
    main() 