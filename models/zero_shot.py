import time
import pandas as pd
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

EXAMPLES = [
    "hardware",
    "artificial intelligence",
    "finance and markets",
    "politics and government",
    "health and medicine",
    "software",
    "science and space",
    "entertainment",
]

class TopicClassifier:
    def __init__(self, topics: list[str] = None, device: int = -1):
        self.topics = topics or EXAMPLES
        print(f"Loading model...")
        print(f"Topics: {self.topics}")
        self.pipe = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
        print("Zero-shot model loaded!!")

    def _prep_text(self, title: str, text: str = "") -> str:
        title = str(title).strip() if pd.notna(title) else ""
        text  = str(text).strip()  if pd.notna(text)  else ""
        if text:
            return text[:300]
        
        return f"{title}. {text}" if text else title
 
    def predict_single(self, title: str, text: str = "", multi_label: bool = False) -> dict:
        articles = self._prep_text(title, text)
        raw = self.pipe(articles, candidate_labels=self.topics, multi_label=multi_label)
        all_scores = {}
        for label, score in zip(raw["labels"], raw["scores"]):
            all_scores[label] = round(score, 4)
        #all_scores = {label: round(score, 4) for label, score in zip(raw["labels"], raw["scores"])}
        return {"topic": raw["labels"][0], "confidence": round(raw["scores"][0], 4), "all_scores": all_scores}
 
    def predict_batch(self, df: pd.DataFrame, title_col: str = "title", text_col: str = "text", batch_size: int = 16, multi_label: bool = False) -> pd.DataFrame:
        texts = []

        for _, row in df.iterrows():
            texts.append(self._prep_text(row[title_col], row.get(text_col, "")))

        print(f"Running zero-shot classification on {len(texts):,} articles...")
        print(f"Note: BART is slower than RoBERTa — expect ~2-4s per article on CPU.")
        start = time.time()
 
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"BATCH {i}: {len(batch)} texts")
            for text in batch:
                print(f" processing: {text[:50]}")
                raw = self.pipe(text, candidate_labels=self.topics, multi_label=multi_label)

                if isinstance(raw, list):
                    raw = raw[0]
                
                all_scores = {}
                for label, score in zip(raw["labels"], raw["scores"]):
                    all_scores[label] = round(score, 4)

                all_results.append({"topic": raw["labels"][0], "topic_confidence": round(raw["scores"][0], 4), "topic_scores": all_scores})
 
            done = min(i + batch_size, len(texts))
            # elapsed = round(time.time() - start, 1)
            # rate = round(done / elapsed, 1) if elapsed > 0 else "—"
            print(f"{done} out of {len(texts)} articles completed...")
 
        print(f"Finished in {round(time.time() - start, 2)} seconds")
        print(all_results)
 
        output = df.copy()
        output["topic"] = [r["topic"] for r in all_results]
        output["topic_confidence"] = [r["topic_confidence"] for r in all_results]
        output["topic_scores"] = [r["topic_scores"] for r in all_results]
 
        return output
 
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print topic distribution after running predict_batch."""
        if "topic" not in df.columns:
            print("Run predict_batch first.")
            return
        counts = df["topic"].value_counts()
        print("\nTopic distribution:")
        print("----------------------------------")
        for topic, n in counts.items():
            pct = round(n / len(df) * 100, 1)
            print(f"{topic}: {n} articles ({pct}%)")

        avg_confidence = df['topic_confidence'].mean()
        print(f"\navg confidence : {avg_confidence:.3f}")
        # low_conf = (df['topic_confidence'] < 0.5).sum()
        # print(f"  low conf (<0.5): {low_conf} articles — consider reviewing these")

 
if __name__ == "__main__":
    clf = TopicClassifier()
 
    test_cases = [
        ("Fed raises interest rates by 25 basis points", "The decision came amid ongoing inflation concerns."),
        ("OpenAI releases new reasoning model", "The model outperforms previous versions on benchmarks."),
        ("Cancer vaccine shows promise in clinical trial", "Researchers report a 90 percent response rate in early tests."),
        ("Ronaldo scores hat-trick in Champions League", "Real Madrid advance to the semi-finals."),
        ("Arctic sea ice hits record low for third year", "Scientists warn of accelerating feedback loops."),
    ]
 
    print("\nSingle prediction test:")
    print("-----------------------------------")
    for title, text in test_cases:
        result = clf.predict_single(title, text)
        print(f"Title : {title}")
        print(f"Topic : {result['topic']} (confidence: {result['confidence']})")
        print()
 
    print("Batch prediction test:")
    print("-----------------------------------")
    test_df = pd.DataFrame(test_cases, columns=["title", "text"])
    print(test_df.columns.tolist())
    print(len(test_df))
    out_df = clf.predict_batch(test_df)
    clf.print_summary(out_df)
    print("\nSample output:")
    print(out_df[["title", "topic", "topic_confidence"]].to_string(index=False))
