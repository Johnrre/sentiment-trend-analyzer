# News Sentiment & Trend Analyzer

Explore how the tone of English-language news coverage changes over time. The project collects articles from NewsAPI, labels each one with a sentiment and a topic, and displays the results in a Streamlit dashboard. For example, select **finance and markets** to see daily counts of positive, neutral, and negative stories, then filter by source to see which outlets contributed to those counts.

## How articles are classified

| Task | Pretrained model | Input and output |
| --- | --- | --- |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Reads the title and available article text. Returns positive, neutral, or negative, with a score for each label. |
| Topic | `facebook/bart-large-mnli` | Reads the first 300 characters of the article text, falling back to the title. Selects the highest-scoring candidate topic using zero-shot classification. |

The eight candidate topics are hardware, artificial intelligence, finance and markets, politics and government, health and medicine, software, science and space, and entertainment. Edit the list in [models/zero_shot.py](models/zero_shot.py) to change them. Both models run on CPU by default and download their weights on first use.

## Set up

Use Python 3.11 and a NewsAPI key. The following commands use Bash and work in WSL. Run them from the same terminal so the project uses the active environment and API key throughout.

```bash
git clone https://github.com/Johnrre/sentiment-trend-analyzer.git
cd sentiment-trend-analyzer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt "numpy<2" jupyterlab matplotlib seaborn "plotly>=5.5,<6" requests schedule
mkdir -p data/processed
read -rsp "Enter your NewsAPI key: " NEWSAPI_KEY
export NEWSAPI_KEY
echo
```

## Run the pipeline

### 1. Collect articles

```bash
python ingestion/news_pull.py
```

The default pull searches for **stocks** and fetches **business** and **technology** headlines. It stores the title, description or content snippet, source, URL, and timestamps in `data/raw/news.db`. A URL-derived primary key prevents the same URL from being inserted twice. To change the queries or categories, edit `topics` in [ingestion/news_pull.py](ingestion/news_pull.py).

### 2. Prepare the modeling data

```bash
jupyter lab notebooks/eda.ipynb
```

Run all cells in order with the notebook's working directory set to `notebooks/`. The notebook reads the SQLite database, removes duplicate titles, and excludes rows with missing text, fewer than five combined words, or a NewsAPI truncation marker. It writes the remaining articles to `data/processed/articles_clean.csv`. Check that the final cell reports at least one article before continuing.

### 3. Generate predictions

**One-time compatibility fix.** In [models/run_inference.py](models/run_inference.py), replace the two lines that assign `low_sent` and `low_topic` with the following, keeping their indentation inside `run()`. The runner expects flag columns that the classifiers do not return. These expressions calculate the counts directly from the scores using the thresholds in the model scripts.

```python
    low_sent = (df["sentiment_score"] < 0.6).sum()
    low_topic = (df["topic_confidence"] < 0.5).sum()
```

From the repository root, try a small run first.

```bash
python models/run_inference.py --limit 10 --no-wandb
```

Then process the full cleaned dataset.

```bash
python models/run_inference.py --no-wandb
```

Each run replaces `data/processed/predictions.csv`. The file contains the article metadata, publication date, predicted labels, and confidence scores. Use `--batch-size N` to change the default group size of 16. Omit `--no-wandb` to enable Weights & Biases logging for runtimes, label distributions, and sample predictions.

### 4. Open the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL printed in the terminal. The dashboard reads the predictions CSV and lets you filter by topic, sentiment, and source. The time series shows **article counts per day**. The sentiment-by-topic heatmap shows **percentages within each topic**. For example, a higher negative count on a busy news day can coincide with a lower negative share.

## Monitor changes between runs

After generating predictions for a representative batch of articles, run the monitor from the repository root.

```bash
python monitoring/drift.py
```

The first run sorts predictions by date and saves the oldest 40% as the baseline in `data/processed/reference.csv`. After refreshing the predictions, run the same command again. Evidently compares the saved baseline with the entire current predictions file, including any overlapping articles. It checks for changes in sentiment labels, topics, and model scores.

Later runs write an HTML report under `monitoring/reports/` and append a result to `monitoring/drift_summary.json`. The dashboard uses that history to show which columns drifted and how the share of drifted columns changed between runs.

## Schedule collection

```bash
python ingestion/scheduler.py
```

Leave the process running to collect articles at **08:00 and 20:00** in the machine's local time. Logs go to `data/scheduler.log`. To display newly collected articles, rerun the preparation notebook and inference. The dashboard caches its inputs for up to 30 minutes.
