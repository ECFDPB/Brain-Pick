import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from emotion_to_tagsreport import EmotionToTagsReportConverter, TagsReport, Tag


class TrainingPredictionsProcessor:
    def __init__(self, csv_path: str, username: str = "EFBPE"):
        self.csv_path = csv_path
        self.converter = EmotionToTagsReportConverter(username=username)
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} records from {self.csv_path}")
        print(f"Columns: {list(self.data.columns)}")

    def convert_to_emotion_result(self, row) -> Dict[str, Any]:
        return {
            "emotion": str(row["Predicted_Label"]).upper(),
            "confidence": float(row["Confidence"]),
            "probabilities": {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            },
        }

    def process_all_predictions(self) -> List[TagsReport]:
        if self.data is None:
            self.load_data()

        base_timestamp = int(time.time())
        tags_reports = []

        for i, (_, row) in enumerate(self.data.iterrows()):
            result = self.convert_to_emotion_result(row)
            tags_reports.append(self.converter.convert_with_probabilities(result, base_timestamp + i))

        return tags_reports

    def process_sample(self, sample_size: int = 10) -> List[TagsReport]:
        if self.data is None:
            self.load_data()

        sample_data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)
        tags_reports = []
        base_timestamp = int(time.time())

        for i, (_, row) in enumerate(sample_data.iterrows()):
            result = self.convert_to_emotion_result(row)
            print(f"\nSample {i + 1}:")
            print(f"  True: {row['True_Label']}, Predicted: {row['Predicted_Label']}, Confidence: {row['Confidence']:.4f}")
            print(f"  Probs: NEG={row['Prob_NEGATIVE']:.4f}, NEU={row['Prob_NEUTRAL']:.4f}, POS={row['Prob_POSITIVE']:.4f}")

            report = self.converter.convert_with_probabilities(result, base_timestamp + i)
            tags_reports.append(report)

            print(f"  likeness: {report.value:.4f}, tags: {len(report.topic)}, top: {[t.name for t in report.topic[:3]]}")

        return tags_reports

    def save_to_json(self, tags_reports: List[TagsReport], output_path: str):
        reports_dict = [self.converter.to_dict(r) for r in tags_reports]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reports_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(tags_reports)} reports to {output_path}")

    def analyze_results(self, tags_reports: List[TagsReport]):
        print("\n" + "=" * 50)
        print("Results Analysis")
        print("=" * 50)

        likeness_values = [r.value for r in tags_reports]
        tag_counts = [len(r.topic) for r in tags_reports]

        print(f"Total reports: {len(tags_reports)}")
        print(f"Likeness range: [{min(likeness_values):.4f}, {max(likeness_values):.4f}]")
        print(f"Mean likeness: {np.mean(likeness_values):.4f}, Std: {np.std(likeness_values):.4f}")
        print(f"Mean tag count: {np.mean(tag_counts):.2f}, Range: [{min(tag_counts)}, {max(tag_counts)}]")

        print("\nLikeness distribution:")
        thresholds = [(-1.0, -0.8, "Very negative"), (-0.8, -0.3, "Moderately negative"),
                      (-0.3, 0.0, "Slightly negative"), (0.0, 0.0, "Neutral"),
                      (0.0, 0.3, "Slightly positive"), (0.3, 0.8, "Moderately positive"), (0.8, 1.0, "Very positive")]
        for lo, hi, label in thresholds:
            count = sum(1 for v in likeness_values if lo <= v <= hi) if lo == hi else sum(1 for v in likeness_values if lo < v <= hi)
            print(f"  {label}: {count}")

        tag_frequency: Dict[str, int] = {}
        for report in tags_reports:
            for tag in report.topic:
                tag_frequency[tag.name] = tag_frequency.get(tag.name, 0) + 1

        print("\nTop 10 tags:")
        for tag_name, count in sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {tag_name}: {count} ({count / len(tags_reports) * 100:.1f}%)")

    def compare_with_true_labels(self):
        if self.data is None:
            self.load_data()

        print("\n" + "=" * 50)
        print("Comparison with True Labels")
        print("=" * 50)

        correct = 0
        for _, row in self.data.iterrows():
            true_label = str(row["True_Label"]).upper()
            probs = {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            }
            likeness = sum(self.converter.EMOTION_TO_LIKENESS.get(k, 0.0) * v for k, v in probs.items())
            likeness = max(-1.0, min(1.0, likeness))

            if likeness <= -0.5:
                mapped = "NEGATIVE"
            elif likeness >= 0.5:
                mapped = "POSITIVE"
            else:
                mapped = "NEUTRAL"

            if mapped == true_label:
                correct += 1

        total = len(self.data)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"Total: {total}, Correct: {correct} ({accuracy:.2f}%)")
        return accuracy


def main():
    processor = TrainingPredictionsProcessor(csv_path="training_set_predictions.csv", username="EFBPE")

    print("1. Sample predictions")
    sample_reports = processor.process_sample(sample_size=5)
    print(f"\nFirst sample JSON:\n{processor.converter.to_json(sample_reports[0])}")

    print("\n2. All predictions")
    all_reports = processor.process_all_predictions()

    processor.analyze_results(all_reports)
    processor.compare_with_true_labels()

    output_path = f"tags_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    processor.save_to_json(all_reports, output_path)

    summary = {
        "processing_date": datetime.now().isoformat(),
        "username": "EFBPE",
        "total_reports": len(all_reports),
        "likeness_statistics": {
            "mean": float(np.mean([r.value for r in all_reports])),
            "std": float(np.std([r.value for r in all_reports])),
            "min": float(min(r.value for r in all_reports)),
            "max": float(max(r.value for r in all_reports)),
        },
        "sample_reports": [processor.converter.to_dict(r) for r in all_reports[:3]],
    }

    summary_path = f"tags_reports_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
