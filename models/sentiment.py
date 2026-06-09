import time
import pandas as pd
from transformers import pipeline, AutoTokenizer

#using RoBERTa for finding the sentiment
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    "positive": "positive",
}

# device = 0 GPU , -1 CPU (default and safer)
class SentimentAnalyzer:
    def __init__(self, device: int = -1):
        print("Loading model..")
        self.pipe = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,      # return scores for ALL labels, not just the top one
            device=device,
            truncation=True,
            max_length=512,
        )
        print("Model loaded.")
 
    def prep_text(self, title: str, text: str) -> str:
        title = str(title).strip() if pd.notna(title) else ""
        text  = str(text).strip()  if pd.notna(text)  else ""
        
        if text:
          return f"{title}. {text}"[:1200]

        return title[:1200]
 
    def predict_single(self, title: str, text: str = "") -> dict:
        articles = self.prep_text(title, text)
        raw = self.pipe(articles)[0]
        scores = {}
        for r in raw:
            scores[LABEL_MAP[r["label"]]] = round(r["score"], 4)
        top_label = max(scores, key=scores.get)

        return {"label": top_label, "score": scores[top_label], "scores": scores}
 
    def predict_batch(self, df: pd.DataFrame, title_col: str = "title", text_col:  str = "text", batch_size: int = 32) -> pd.DataFrame:
        texts = []

        for _, row in df.iterrows():
            texts.append(self.prep_text(row['title'], row.get("text", "")))

        start = time.time()
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            from_batch = self.pipe(batch)
            for raw in from_batch:
                scores = {}
                for r in raw:
                    scores[LABEL_MAP[r["label"]]] = round(r["score"], 4)                
                top_label = max(scores, key=scores.get)
                results.append({"sentiment_label":  top_label, "sentiment_score":  scores[top_label], "sentiment_scores": scores})
 
            # Progress log every 5 batches
            if (i // batch_size) % 5 == 0:
                print(f"{min(i + batch_size, len(texts))} / {len(texts):,} articles processed...")

        print(f"Finished in {round(time.time() - start, 2)} seconds")
 
        output = df.copy()
        output["sentiment_label"] = [r["sentiment_label"]  for r in results]
        output["sentiment_score"] = [r["sentiment_score"]  for r in results]
        output["sentiment_scores"] = [r["sentiment_scores"] for r in results]
 
        return output
 
    def print_summary(self, df: pd.DataFrame) -> None:
        print("\nRunning summary")
        counts = df["sentiment_label"].value_counts()

        total = len(df)
        print("\nSentiment distribution:")
        print("---------------------------")

        for label in ["positive", "neutral", "negative"]:
            count = counts.get(label, 0)
            pct = round(count / total * 100, 1)
            print(f"{label}: {count} articles ({pct}%)")

        print(f"\navg confidence: {df['sentiment_score'].mean():.3f}")
        # print(f"  low confidence (<0.6): "
        #       f"{(df['sentiment_score'] < 0.6).sum()} articles")

 

#test run!!!
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
 
    test_cases = [("Markets plunge as recession fears grow", "Investors panic as key indicators flash red."),
        ("AI startup raises $500M in record funding", "The round values the company at $4 billion."),
        ("Government releases new climate policy draft", "Details of the proposal remain under review."),
    ]
 
    print("\nSingle prediction test:")
    print("---------------------------")
    for title, text in test_cases:
        result = analyzer.predict_single(title, text)
        print(f"  Title : {title}")
        print(f"  Result: {result['label']} (confidence: {result['score']})")
        print(f"  Scores: {result['scores']}")
 
    print("\nBatch prediction test:")
    print("---------------------------")
    test_df = pd.DataFrame(test_cases, columns=["title", "text"])
    out_df = analyzer.predict_batch(test_df)
    analyzer.print_summary(out_df)
    print("\nSample output:")
    print(out_df[["title", "sentiment_label", "sentiment_score"]].to_string(index=False))
