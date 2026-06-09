import argparse
import logging
import time
import json
import pathlib
import pandas as pd
import wandb
from sentiment import SentimentAnalyzer
from zero_shot import TopicClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_PATH  = "./data/processed/articles_clean.csv"
OUTPUT_PATH = "./data/processed/predictions.csv"

def init_wandb(config: dict):
    try:
        run = wandb.init(project="news-sentiment-analyzer", name=f"inference-{time.strftime('%Y%m%d-%H%M')}", config=config, tags=["inference", "roberta", "bart"])
        log.info("W&B run initialised.")
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({e}). Continuing without tracking.")
        return None

def log_metrics(run, metrics: dict):
    if run is None:
        return
    try:
        wandb.log(metrics)
    except Exception:
        pass

def finish_wandb(run, df: pd.DataFrame):
    if run is None:
        return
    
    try:
        # Log sentiment distribution as a bar chart
        sentiment_data = []
        topic_data = []
        label_counts = df["sentiment_label"].value_counts().to_dict()

        for k, v in label_counts.items():
            sentiment_data.append([k,v])
        for k, v in df["topic"].value_counts().to_dict().items():
            topic_data.append([k, v])
        
        wandb.log({
            "sentiment_distribution": wandb.plot.bar(wandb.Table(
            data=sentiment_data,
            columns=["label", "count"]),
            "label", "count", title="Sentiment Distribution"
            ), 
            "topic_distribution": wandb.plot.bar(wandb.Table(
            data=topic_data,
            columns=["topic", "count"]),
            "topic", "count", title="Topic Distribution"
            )
        })
        
        # Log a sample of predictions as a table
        sample = df[["title", "sentiment_label", "sentiment_score", "topic", "topic_confidence"]].head(50)
        wandb.log({"predictions_sample": wandb.Table(dataframe=sample)})
        run.finish()
        log.info("W&B run finished.")

    except Exception as e:
        log.warning(f"W&B finish failed: {e}")


def run(limit: int = None, use_wandb: bool = True, batch_size: int = 16):
    log.info(f"Loading data from {INPUT_PATH}...")
    if not pathlib.Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}. Make sure you've run the EDA notebook and saved articles_clean.csv first.")

    df = pd.read_csv(INPUT_PATH)
    log.info(f"Loaded {len(df):,} articles.")

    if limit:
        df = df.head(limit)
        log.info(f"Limit set — using first {limit} rows for this run.")

    config = {"sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
              "topic_model": "facebook/bart-large-mnli",
              "n_articles": len(df),
              "batch_size": batch_size,
              "limit": limit,
            }

    if use_wandb:
        run_obj = init_wandb(config)
    else:
        run_obj = None

    log.info("----------------------------------------------------------------")
    log.info("STEP 1/2 — Sentiment analysis (RoBERTa)")
    log.info("----------------------------------------------------------------")

    sentiment_analyzer = SentimentAnalyzer()

    start = time.time()
    df = sentiment_analyzer.predict_batch(df, batch_size=batch_size)
    sentiment_time = round(time.time() - start, 1)

    sentiment_analyzer.print_summary(df)
    log_metrics(run_obj, {"sentiment_runtime_s": sentiment_time,
                          "sentiment_avg_score": round(df["sentiment_score"].mean(), 4),
                          "sentiment_pct_positive": round((df["sentiment_label"] == "positive").mean(), 4),
                          "sentiment_pct_neutral": round((df["sentiment_label"] == "neutral").mean(), 4),
                          "sentiment_pct_negative": round((df["sentiment_label"] == "negative").mean(), 4),
                        })

    del sentiment_analyzer
    log.info("RoBERTa model unloaded from memory.")

    log.info("----------------------------------------------------------------")
    log.info("STEP 2/2 — Topic classification (BART zero-shot)")
    log.info("----------------------------------------------------------------")

    topic_classifier = TopicClassifier()

    start = time.time()
    df = topic_classifier.predict_batch(df, batch_size=batch_size)
    topic_time = round(time.time() - start, 1)

    topic_classifier.print_summary(df)
    log_metrics(run_obj, {"topic_runtime_s": topic_time, 
                          "topic_avg_confidence": round(df["topic_confidence"].mean(), 4), 
                          "top_topic": df["topic"].value_counts().idxmax()
                        })

    del topic_classifier
    log.info("Post-processing predictions...")

    if "sentiment_scores" in df.columns:
        scores_df = df["sentiment_scores"].apply(lambda x: x if isinstance(x, dict) else json.loads(x)).apply(pd.Series).add_prefix("sent_score_")
        df = pd.concat([df.drop(columns=["sentiment_scores"]), scores_df], axis=1)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date

    low_sent = df["sentiment_low_conf"].sum()
    low_topic = df["topic_low_conf"].sum()

    pathlib.Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Predictions saved to {OUTPUT_PATH}")

    log_metrics(run_obj, {"total_runtime_s": round(sentiment_time + topic_time, 1), 
                          "low_conf_sentiment": int(low_sent), 
                          "low_conf_topic": int(low_topic),
                          "output_rows": len(df),
                          "output_cols": len(df.columns),
                        })
    finish_wandb(run_obj, df)

    log.info("----------------------------------------------------------------")
    log.info("ANALYSIS DONE!")
    log.info(f"Articles processed: {len(df):,}")
    log.info(f"Sentiment runtime: {sentiment_time}s")
    log.info(f"Topic runtime: {topic_time}s")
    log.info(f"Output: {OUTPUT_PATH}")
    log.info("----------------------------------------------------------------")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full inference pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for both models")
    args = parser.parse_args()

    run(limit=args.limit, use_wandb=not args.no_wandb, batch_size=args.batch_size)