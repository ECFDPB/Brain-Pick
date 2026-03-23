import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any


def map_to_likeness(probs: Dict[str, float]) -> float:
    """Weighted sum: -1*P(neg) + 0*P(neu) + 1*P(pos)"""
    value = -1.0 * probs.get("NEGATIVE", 0) + 1.0 * probs.get("POSITIVE", 0)
    return max(-1.0, min(1.0, value))


class TrainingPredictionsProcessor:
    def __init__(self, csv_path: str, username: str = "EFBPE"):
        self.csv_path = csv_path
        self.username = username
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} records from {self.csv_path}")

    def _row_to_result(self, row) -> Dict[str, Any]:
        return {
            "emotion": str(row["Predicted_Label"]).upper(),
            "confidence": float(row["Confidence"]),
            "probabilities": {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            },
        }

    def process_all(self) -> List[Dict]:
        if self.data is None:
            self.load_data()

        base_timestamp = int(time.time())
        reports = []
        for i, (_, row) in enumerate(self.data.iterrows()):
            result = self._row_to_result(row)
            reports.append({
                "username": self.username,
                "timestamp": base_timestamp + i,
                "emotion": result["emotion"],
                "value": map_to_likeness(result["probabilities"]),
            })
        return reports

    def process_sample(self, sample_size: int = 10) -> List[Dict]:
        if self.data is None:
            self.load_data()

        sample_data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)
        reports = []
        base_timestamp = int(time.time())

        for i, (_, row) in enumerate(sample_data.iterrows()):
            result = self._row_to_result(row)
            value = map_to_likeness(result["probabilities"])
            print(f"\nSample {i + 1}:")
            print(f"  True: {row['True_Label']}, Predicted: {row['Predicted_Label']}, Confidence: {row['Confidence']:.4f}")
            print(f"  likeness: {value:.4f}")
            reports.append({
                "username": self.username,
                "timestamp": base_timestamp + i,
                "emotion": result["emotion"],
                "value": value,
            })
        return reports

    def save_to_json(self, reports: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(reports)} reports to {output_path}")

    def analyze(self, reports: List[Dict]):
        print("\n" + "=" * 50)
        values = [r["value"] for r in reports]
        print(f"Total: {len(reports)}")
        print(f"Likeness range: [{min(values):.4f}, {max(values):.4f}]")
        print(f"Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

        buckets = [
            (lambda v: v <= -0.5, "Negative (≤ -0.5)"),
            (lambda v: -0.5 < v < 0.5, "Neutral (-0.5, 0.5)"),
            (lambda v: v >= 0.5, "Positive (≥ 0.5)"),
        ]
        for fn, label in buckets:
            count = sum(1 for v in values if fn(v))
            print(f"  {label}: {count} ({count / len(values) * 100:.1f}%)")

    def compare_with_true_labels(self) -> float:
        if self.data is None:
            self.load_data()

        correct = 0
        for _, row in self.data.iterrows():
            true_label = str(row["True_Label"]).upper()
            probs = {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            }
            likeness = map_to_likeness(probs)
            mapped = "NEGATIVE" if likeness <= -0.5 else ("POSITIVE" if likeness >= 0.5 else "NEUTRAL")
            if mapped == true_label:
                correct += 1

        total = len(self.data)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"Mapping accuracy: {correct}/{total} = {accuracy:.2f}%")
        return accuracy


def main():
    processor = TrainingPredictionsProcessor(csv_path="training_set_predictions.csv", username="EFBPE")

    print("Sample predictions:")
    processor.process_sample(sample_size=5)

    print("\nAll predictions:")
    all_reports = processor.process_all()
    processor.analyze(all_reports)
    processor.compare_with_true_labels()

    output_path = f"tags_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    processor.save_to_json(all_reports, output_path)


if __name__ == "__main__":
    main()
