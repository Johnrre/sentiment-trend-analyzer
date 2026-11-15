import json
import pathlib
import pandas as pd
import streamlit as st
import plotly.express as px

PREDICTIONS_PATH = "data/processed/predictions.csv"
DRIFT_SUMMARY_PATH = "monitoring/drift_summary.json"
DRIFT_REPORTS_DIR = "monitoring/reports"

st.set_page_config(page_title="News Sentiment Trend Analyzer", page_icon="📊", layout="wide")

@st.cache_data(ttl=1800)
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    #if "published_at" in df.columns:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    #if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

@st.cache_data(ttl=1800)
def load_drift_summary() -> list[dict]:
    path = pathlib.Path(DRIFT_SUMMARY_PATH)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def get_latest_drift_report() -> pathlib.Path | None:
    reports_dir = pathlib.Path(DRIFT_REPORTS_DIR)
    if not reports_dir.exists():
        return None
    reports = sorted(reports_dir.glob("drift_*.html"))
    return reports[-1] if reports else None

try:
    df = load_predictions()
except FileNotFoundError:
    st.error(f"Could not find {PREDICTIONS_PATH}. Run `python models/run_inference.py` first.")
    st.stop()

drift_history = load_drift_summary()
st.title("News Sentiment & Trend Analyzer")
st.caption("End-to-end ML pipeline: live news ingestion → HuggingFace sentiment & topic classification → Evidently drift monitoring")

st.sidebar.header("Filters")

#if "topic" in df.columns:
topics = ["All"] + sorted(df["topic"].dropna().unique().tolist())
selected_topic = st.sidebar.selectbox("Topic", topics)
if selected_topic != "All":
    df = df[df["topic"] == selected_topic]

#if "sentiment_label" in df.columns:
sentiments = ["All"] + sorted(df["sentiment_label"].dropna().unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
if selected_sentiment != "All":
    df = df[df["sentiment_label"] == selected_sentiment]

#if "source" in df.columns:
sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Source", sources)
if selected_source != "All":
    df = df[df["source"] == selected_source]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of total articles after filters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total articles", f"{len(df):,}")

with col2:
    if "sentiment_label" in df.columns and len(df) > 0:
        pct_positive = (df["sentiment_label"] == "positive").mean()
        st.metric("Positive sentiment", f"{pct_positive:.1%}")
    else:
        st.metric("Positive sentiment", "—")

with col3:
    if "sentiment_score" in df.columns and len(df) > 0:
        st.metric("Avg sentiment confidence", f"{df['sentiment_score'].mean():.2f}")
    else:
        st.metric("Avg sentiment confidence", "—")

with col4:
    if "topic_confidence" in df.columns and len(df) > 0:
        st.metric("Avg topic confidence", f"{df['topic_confidence'].mean():.2f}")
    else:
        st.metric("Avg topic confidence", "—")

st.markdown("---")

st.subheader("Sentiment over time")

if "date" in df.columns and "sentiment_label" in df.columns and len(df) > 0:
    trend = (df.groupby([df["date"].dt.date, "sentiment_label"]).size().reset_index(name="count").rename(columns={"date": "date"}))
    trend.columns = ["date", "sentiment_label", "count"]

    fig = px.line(trend, x="date", y="count", color="sentiment_label", markers=True, 
                  color_discrete_map={"positive": "#1D9E75", "neutral": "#9CA3AF", "negative": "#D85A30", },
                  labels={"count": "Article count", "date": "Date", "sentiment_label": "Sentiment"}
                  )
    fig.update_layout(legend_title_text="Sentiment", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data to show sentiment trends yet.")


col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Topic distribution")
    if "topic" in df.columns and len(df) > 0:
        topic_counts = df["topic"].value_counts().reset_index()
        topic_counts.columns = ["topic", "count"]
        fig = px.bar(topic_counts, x="count", y="topic", orientation="h", color="count", color_continuous_scale="Blues")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No topic data available.")

with col_right:
    st.subheader("Sentiment breakdown")
    if "sentiment_label" in df.columns and len(df) > 0:
        sent_counts = df["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig = px.pie(sent_counts, names="sentiment", values="count", color="sentiment", 
                     color_discrete_map={"positive": "#1D9E75", "neutral": "#9CA3AF", "negative": "#D85A30"},
                     hole=0.4,
                    )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available.")

st.subheader("Sentiment by topic")

if "topic" in df.columns and "sentiment_label" in df.columns and len(df) > 0:
    cross = pd.crosstab(df["topic"], df["sentiment_label"], normalize="index").round(3)
    # Reorder columns if all present
    col_order = []
    for c in ["positive", "neutral", "negative"]:
        if c in cross.columns:
            col_order.append(c)
#    col_order = [c for c in ["positive", "neutral", "negative"] if c in cross.columns]
    cross = cross[col_order]

    fig = px.imshow(cross, text_auto=".0%", color_continuous_scale="RdYlGn", aspect="auto", labels=dict(x="Sentiment", y="Topic", color="Share"))
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data for sentiment-by-topic breakdown.")

st.markdown("---")

st.subheader("Model drift monitoring")

if drift_history:
    # Drift score over time
    drift_df = pd.DataFrame([{"run": i + 1, "timestamp": entry["timestamp"], "share_drifted": entry.get("share_drifted", 0), 
                              "n_drifted_columns": entry.get("n_drifted_columns", 0),
                              "dataset_drifted": entry.get("dataset_drifted", False)
                              }
                            for i, entry in enumerate(drift_history)]
                            )

    drift_col1, drift_col2 = st.columns([2, 1])

    with drift_col1:
        fig = px.bar(drift_df, x="run", y="share_drifted", color="dataset_drifted", color_discrete_map={True: "#D85A30", False: "#1D9E75"},
                     labels={"share_drifted": "Share of drifted columns", "run": "Monitoring run", "dataset_drifted": "Drift detected"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with drift_col2:
        latest = drift_history[-1]
        if latest.get("dataset_drifted"):
            st.error("!!Drift detected in latest run")
        else:
            st.success("Yay! No drift in latest run")

        st.metric("Drifted columns", latest.get("n_drifted_columns", 0))
        st.metric("Share drifted", f"{latest.get('share_drifted', 0):.1%}")

        if latest.get("drifted_columns"):
            st.write("**Drifted columns:**")
            for col in latest["drifted_columns"]:
                score = latest["column_drift"].get(col, {}).get("score", "—")
                st.write(f"- `{col}` (score: {score})")

else:
    st.info(
        "No drift reports found yet. Run `python monitoring/drift.py` "
        "to generate your first drift report."
    )

st.markdown("---")
st.caption(
    "Built with HuggingFace Transformers (RoBERTa sentiment + BART zero-shot topics) "
    "and Evidently AI for drift monitoring."
)