import pickle
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import os


@dataclass
class TagsReport:
    username: str
    timestamp: int
    value: float  # -1.0 to 1.0, represents likeness


def map_prediction_to_value(prediction, prediction_proba=None):
    """Map model prediction to [-1.0, 1.0] range using probability weighting."""
    if prediction_proba is not None:
        if len(prediction_proba) >= 3:
            value = -1.0 * prediction_proba[0] + 0.0 * prediction_proba[1] + 1.0 * prediction_proba[2]
            return max(-1.0, min(1.0, value))
        elif len(prediction_proba) == 2:
            value = -1.0 * prediction_proba[0] + 1.0 * prediction_proba[1]
            return max(-1.0, min(1.0, value))

    mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    return mapping.get(prediction, max(-1.0, min(1.0, (prediction - 1) / 1.0)))


def generate_timestamps(n_samples, start_time=None, interval_seconds=1):
    if start_time is None:
        start_time = int(time.time())
    return [start_time + i * interval_seconds for i in range(n_samples)]


def load_predictions_from_model(data_path="emotions.csv"):
    try:
        with open("emotion_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        data = pd.read_csv(data_path)
        if "label" in data.columns:
            features = data.drop("label", axis=1).values
            true_labels = data["label"].values
        else:
            features = data.values
            true_labels = None

        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        prediction_probas = model.predict_proba(features_scaled) if hasattr(model, "predict_proba") else None

        print(f"Predicted {len(predictions)} samples")
        return predictions, prediction_probas, true_labels

    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None, None


def create_tags_reports(predictions, prediction_probas=None, username_prefix="user", start_timestamp=None):
    n_samples = len(predictions)
    timestamps = generate_timestamps(n_samples, start_timestamp)

    return [
        TagsReport(
            username=f"{username_prefix}_{i:04d}",
            timestamp=timestamps[i],
            value=float(map_prediction_to_value(
                predictions[i],
                prediction_probas[i] if prediction_probas is not None else None,
            )),
        )
        for i in range(n_samples)
    ]


def export_to_csv(tags_reports, output_file="model_predictions.csv"):
    df = pd.DataFrame([asdict(r) for r in tags_reports])[["username", "timestamp", "value"]]
    df.to_csv(output_file, index=False)

    print(f"Exported {len(tags_reports)} records to {output_file}")
    print(df.head())
    total = len(df)
    negative = df[df["value"] < -0.33].shape[0]
    neutral = df[(df["value"] >= -0.33) & (df["value"] <= 0.33)].shape[0]
    positive = df[df["value"] > 0.33].shape[0]
    print(f"Negative: {negative} ({negative/total*100:.1f}%), Neutral: {neutral} ({neutral/total*100:.1f}%), Positive: {positive} ({positive/total*100:.1f}%)")
    return df


def export_detailed_report(tags_reports, predictions, true_labels=None, output_file="detailed_predictions.csv"):
    class_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    data = []

    for i, report in enumerate(tags_reports):
        row = asdict(report)
        row["prediction"] = int(predictions[i])
        row["prediction_label"] = class_labels.get(predictions[i], str(predictions[i]))

        if true_labels is not None and i < len(true_labels):
            row["true_label"] = true_labels[i]
            row["correct"] = class_labels.get(predictions[i]) == true_labels[i]

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Detailed report saved: {output_file}")

    if true_labels is not None and "correct" in df.columns:
        print(f"Accuracy: {df['correct'].mean():.2%}")

    return df


def create_sample_format():
    sample_data = [
        TagsReport(username="user_0001", timestamp=1672531200, value=-0.8),
        TagsReport(username="user_0002", timestamp=1672531201, value=0.0),
        TagsReport(username="user_0003", timestamp=1672531202, value=0.9),
        TagsReport(username="user_0004", timestamp=1672531203, value=-0.2),
        TagsReport(username="user_0005", timestamp=1672531204, value=1.0),
    ]
    df = pd.DataFrame([asdict(r) for r in sample_data])
    df.to_csv("sample_format.csv", index=False)
    print("Sample format saved to sample_format.csv")
    print(df.to_string(index=False))


def batch_export(data_files, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    for data_file in data_files:
        if os.path.exists(data_file):
            predictions, prediction_probas, _ = load_predictions_from_model(data_file)
            if predictions is not None:
                tags_reports = create_tags_reports(predictions, prediction_probas)
                base_name = os.path.splitext(os.path.basename(data_file))[0]
                export_to_csv(tags_reports, os.path.join(output_dir, f"{base_name}_predictions.csv"))
    print(f"Batch export complete. Results in {output_dir}/")


def main():
    if not os.path.exists("emotion_model.pkl"):
        print("Model file emotion_model.pkl not found. Run train_model.py first.")
        return

    data_files = ["test_data.csv", "emotions_train.csv", "simple_test.csv"]
    data_file = next((f for f in data_files if os.path.exists(f)), None)

    if data_file is None:
        print("No data file found.")
        return

    predictions, prediction_probas, true_labels = load_predictions_from_model(data_file)
    if predictions is None:
        return

    tags_reports = create_tags_reports(predictions, prediction_probas)
    export_to_csv(tags_reports, "model_predictions.csv")
    export_detailed_report(tags_reports, predictions, true_labels, "detailed_predictions.csv")
    create_sample_format()


if __name__ == "__main__":
    main()
