import logging
import pathlib
import json
import pandas as pd
import evidently
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (ColumnDriftMetric, DatasetDriftMetric)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PREDICTIONS_PATH  = "data/processed/predictions.csv"
REFERENCE_PATH    = "data/processed/reference.csv"
REPORT_DIR        = "monitoring/reports"
SUMMARY_PATH      = "monitoring/drift_summary.json"

SENTIMENT_SCORE_COLS = ["sent_score_positive", "sent_score_neutral", "sent_score_negative"]
NUMERIC_COLS = SENTIMENT_SCORE_COLS + ["sentiment_score", "topic_confidence"]
CATEGORICAL_COLS = ["sentiment_label", "topic"]


class DriftMonitor:
    def __init__(self):
        self.check_evidently()

    def check_evidently(self):
        try:
            log.info(f"Evidently version: {evidently.__version__}")
        except ImportError:
            raise ImportError("Evidently not installed. Run: pip install evidently")

    def save_reference(self, df: pd.DataFrame) -> None:
        ref = self.select_columns(df)
        pathlib.Path(REFERENCE_PATH).parent.mkdir(parents=True, exist_ok=True)
        ref.to_csv(REFERENCE_PATH, index=False)
        log.info(f"Reference dataset saved: {len(ref)} rows → {REFERENCE_PATH}")

    def load_reference(self) -> pd.DataFrame:
        if not pathlib.Path(REFERENCE_PATH).exists():
            raise FileNotFoundError(f"No reference dataset found at {REFERENCE_PATH}. Run save_reference() first on your initial predictions.")
        return pd.read_csv(REFERENCE_PATH)

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns Evidently will monitor."""
        available = []
        for c in NUMERIC_COLS + CATEGORICAL_COLS:
            available.append(c)
        return df[available].copy()

    def run(self, reference_df: pd.DataFrame, current_df:   pd.DataFrame, save_html:    bool = True) -> dict:
        log.info("Building Evidently drift report...")
        log.info(f"Reference: {len(reference_df)} rows")
        log.info(f"Current: {len(current_df)} rows")

        ref = self.select_columns(reference_df)
        curr = self.select_columns(current_df)

        # Build report with multiple metric presets
        metrics = [DatasetDriftMetric(), DataDriftPreset()]

        # Add per-column drift for our most important columns
        for col in NUMERIC_COLS + CATEGORICAL_COLS:
            if col in ref.columns:
                metrics.append(ColumnDriftMetric(column_name=col))

        report = Report(metrics=metrics)
        report.run(reference_data=ref, current_data=curr)

        # Save HTML report
        if save_html:
            pathlib.Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            report_path = f"{REPORT_DIR}/drift_{timestamp}.html"
            report.save_html(report_path)
            log.info(f"HTML report saved → {report_path}")

        # Extract summary
        summary = self.extract_summary(report, current_df)
        self.save_summary(summary)
        #self._print_summary(summary)

        return summary

    def extract_summary(self, report, current_df: pd.DataFrame) -> dict:
        result = report.as_dict()
        metrics = result.get("metrics", [])

        summary = {"timestamp": datetime.now().isoformat(), 
                   "current_n_rows": len(current_df), 
                   "dataset_drifted": False, 
                   "n_drifted_columns": 0,
                   "drifted_columns": [],
                   "column_drift": {},
                  }

        for m in metrics:
            metric_id = m.get("metric", "")
            result_ = m.get("result", {})

            if "DatasetDriftMetric" in metric_id:
                summary["dataset_drifted"] = result_.get("dataset_drift", False)
                summary["n_drifted_columns"] = result_.get("number_of_drifted_columns", 0)
                summary["share_drifted"] = round(result_.get("share_of_drifted_columns", 0), 3)

            if "ColumnDriftMetric" in metric_id:
                col = result_.get("column_name", "unknown")
                drift = result_.get("drift_detected", False)
                score = round(result_.get("drift_score", 0), 4)
                summary["column_drift"][col] = {"drifted": drift, "score":   score}

                if drift:
                    summary["drifted_columns"].append(col)

        # Add current sentiment distribution for easy comparison
        if "sentiment_label" in current_df.columns:
            counts = current_df["sentiment_label"].value_counts(normalize=True).round(3)
            summary["current_sentiment_dist"] = counts.to_dict()

        if "topic" in current_df.columns:
            counts = current_df["topic"].value_counts(normalize=True).round(3)
            summary["current_topic_dist"] = counts.to_dict()

        return summary

    def save_summary(self, summary: dict) -> None:
        pathlib.Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)

        history = []
        if pathlib.Path(SUMMARY_PATH).exists():
            with open(SUMMARY_PATH) as f:
                history = json.load(f)

        history.append(summary)

        with open(SUMMARY_PATH, "w") as f:
            json.dump(history, f, indent=2)

        log.info(f"Summary appended → {SUMMARY_PATH}")


def split_reference_current(df: pd.DataFrame, date_col: str = "date", split_percentile: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col)
    split_idx = int(len(df) * split_percentile)
    reference_df = df.iloc[:split_idx].copy()
    current_df = df.iloc[split_idx:].copy()
    log.info(f"Split: {len(reference_df)} reference rows, {len(current_df)} current rows")
    return reference_df, current_df


if __name__ == "__main__":
    log.info(f"Loading predictions from {PREDICTIONS_PATH}...")
    df = pd.read_csv(PREDICTIONS_PATH)
    log.info(f"Loaded {len(df):,} rows.")

    ref_path = pathlib.Path(REFERENCE_PATH)

    if not ref_path.exists():
        # First run — split current data chronologically
        log.info("No reference dataset found. Creating a reference dataset instead...")
        reference_df, current_df = split_reference_current(df)

        # Save the reference slice for all future runs
        monitor = DriftMonitor()
        monitor.save_reference(reference_df)
    else:
        # Subsequent runs — use saved reference vs full current predictions
        monitor = DriftMonitor()
        reference_df = monitor.load_reference()
        current_df = df

        summary = monitor.run(reference_df, current_df)

        if summary["dataset_drifted"]:
            print("!WARNING: Drift detected — consider re-running inference on fresh data.")
        else:
            print("No drift detected. Model predictions look stable.")