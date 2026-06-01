import os
import sqlite3
import datetime
import requests
import hashlib
from pathlib import Path

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "9abbb9dd27014aeea4e43ef7332daa13")
BASE_URL = "https://newsapi.org/v2"


def fetch_top_headlines(query: str = None, category: str = "business", country: str = None, page_size: int = 100) -> list[dict]:
    # possible categories: business, entertainment, general, health, science, sports, technology
    params = {"apiKey": NEWSAPI_KEY, "pageSize": page_size, "language": "en",}

    if query:
        params["q"] = query
    if country:
       params["country"] = country
       params["category"] = category
    else:
        params["category"] = category

    resp = requests.get(f"{BASE_URL}/top-headlines", params=params, timeout=10)
    resp.raise_for_status()

    '''do you wanna return the no. of articles???'''
    #no_of_articles = resp.json().get("totalResults")
    articles = resp.json().get("articles", [])
    return _clean_articles(articles)


    #date format - YYYY-MM-DD    sort_by => relevancy / popularity / publishedAt
def fetch_everything(query: str, from_date: str = None, sort_by: str = "publishedAt", page_size: int = 100) -> list[dict]:
    """
    Search all articles across all sources.
    Good for tracking a specific topic over time.
    """
    params = {"apiKey": NEWSAPI_KEY, "q": query, "sortBy": sort_by, "language": "en", "pageSize": page_size}

    if from_date is None:
        params["from_date"] = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
 
    resp = requests.get(f"{BASE_URL}/everything", params=params, timeout=10)
    resp.raise_for_status()

    '''do you wanna return the no. of articles???'''
    #no_of_articles = resp.json().get("totalResults")
    articles = resp.json().get("articles", [])
    return _clean_articles(articles)


def _clean_articles(raw: list[dict]) -> list[dict]:
    """Normalize raw NewsAPI response into clean dicts."""
    cleaned = []
    for a in raw:
        if not a.get("title") or a["title"] == "[Removed]":
            continue
        cleaned.append({
            "id": _make_id(a.get("url", "")),
            "source": a.get("source", {}).get("name", "unknown"),
            "author": a.get("author") or "unknown",
            "title": a.get("title", "").strip(),
            "text": a.get("description") or a.get("content") or "",
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "fetched_at": datetime.datetime.now(datetime.UTC).isoformat()  #this now includes the time zone as well
        })
    return cleaned

def _make_id(url: str) -> str:
    """Stable short ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]

#create the database
 
def init_db(db_path: str = "./data/raw/news.db") -> sqlite3.Connection:
    """Create the articles table if it doesn't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
 
    cursor = sqlite3.connect(db_path)
    cursor.execute("""CREATE TABLE IF NOT EXISTS articles (
                id           TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                text         TEXT,
                source       TEXT,
                author       TEXT,
                url          TEXT,
                published_at TEXT,
                fetched_at   TEXT
                )
            """)
    
    cursor.commit()
    return cursor


def save_articles(articles: list[dict], db_path: str = "./data/raw/news.db") -> int:
    """
    Insert articles into SQLite. Skips duplicates (INSERT OR IGNORE).
    Returns number of new rows inserted.
    """
    conn = init_db(db_path)
    cursor = conn.executemany("""INSERT OR IGNORE INTO articles
            (id, title, text, source, author, url, published_at, fetched_at)
            VALUES (:id, :title, :text, :source, :author, :url, :published_at, :fetched_at)""",
          articles
          )
    conn.commit()
    new_rows = cursor.rowcount
    conn.close()
    return new_rows


def run_daily_pull(db_path: str = "./data/raw/news.db"):

  #place ur required topics here    |    The format is ("category", "query")
  topics = [(None, "stocks"), ("business", None), ("technology", None)]
  total_rows = 0
  for category, query in topics:
    try:
      if query:
        articles = fetch_everything(query = query)
      else:
        articles = fetch_top_headlines(category=category)
      
      rows = save_articles(articles=articles, db_path = db_path)
      total_rows += rows
      if not query:
         query = ""
      if not category:
         category = ""
      print(f"Successfully fetched {len(articles)} {query} {category} articles")
      print(f"{rows} {query} {category} saved!")

    except requests.HTTPError as e:
      print(f" [{query} {category}] HTTP error : {e}")

    except Exception as e:
      print(f"  [{query} {category}] unexpected error: {e}")

  print(f"\nDone, {total_rows} new articles saved to {db_path}")
  return total_rows

      
if __name__ == "__main__":

  database_path = "./data/raw/news.db"

  run_daily_pull(database_path)
  
  # topics = []

  # articles = fetch_everything(query = "stock")
  # print(f"Feteched {len(articles)} articles successfully\n")
  # if articles:
  #   for a in articles:
  #      print(a["title"])
  # else:
  #    print("ERRRRORRR")

  # connection = init_db()
  # if connection:
  #    print(f"Yay there's a connection {connection}\n")
  # else:
  #    print("noooooo")

  # check = input("do you wanna save in the db? (Y/N): ")
  # if check == 'Y':
  #   print("OK")
  #   count = save_articles(articles= articles)
  #   print(f"{count} articles added to the db")


