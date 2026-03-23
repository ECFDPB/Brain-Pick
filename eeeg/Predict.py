import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time
from dataclasses import dataclass, asdict

model = load_model("best_model_feature_csv.keras")
scaler = joblib.load("scaler_fixed.pkl")

CLASS_LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL_TO_IDX = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


@dataclass
class TagsReport:
    username: str
    timestamp: int
    value: float  # -1.0 to 1.0, represents likeness


def predict_from_csv(csv_path, has_label=True):
    df = pd.read_csv(csv_path)
    if has_label:
        return df.iloc[:, :-1].values, df.iloc[:, -1].values, df
    return df.values, None, df


def batch_predict(features_array):
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


def map_prediction_to_value(pred_class, pred_proba):
    if len(pred_proba) >= 3:
        value = -1.0 * pred_proba[0] + 0.0 * pred_proba[1] + 1.0 * pred_proba[2]
        return max(-1.0, min(1.0, value))
    return {0: -1.0, 1: 0.0, 2: 1.0}.get(pred_class, 0.0)


def create_tags_report_from_predictions(predicted_classes, predicted_probas, username_prefix="user", start_timestamp=None):
    if start_timestamp is None:
        start_timestamp = int(time.time())

    return [
        TagsReport(
            username=f"{username_prefix}_{i:04d}",
            timestamp=start_timestamp + i,
            value=float(map_prediction_to_value(pred_class, pred_proba)),
        )
        for i, (pred_class, pred_proba) in enumerate(zip(predicted_classes, predicted_probas))
    ]


def export_tags_report_to_csv(tags_reports, output_file="tags_report.csv"):
    df = pd.DataFrame([asdict(r) for r in tags_reports])[["username", "timestamp", "value"]]
    df.to_csv(output_file, index=False)
    return df


csv_path = "emotions 2.csv"
print(f"Loading: {csv_path}")

features, true_labels, original_df = predict_from_csv(csv_path, has_label=True)
print(f"Features: {features.shape}, Labels: {len(true_labels)}")

predicted_classes, predicted_probas = batch_predict(features)

correct = sum(
    1 for i in range(len(true_labels))
    if (LABEL_TO_IDX.get(true_labels[i], -1) if isinstance(true_labels[i], str) else int(true_labels[i])) == predicted_classes[i]
)
total = len(true_labels)
accuracy = correct / total * 100

print(f"\nResults: {correct}/{total} = {accuracy:.2f}%")
print("\nPer-class accuracy:")

for label_name, label_idx in LABEL_TO_IDX.items():
    mask = [
        (LABEL_TO_IDX.get(l, -1) == label_idx if isinstance(l, str) else l == label_idx)
        for l in true_labels
    ]
    if any(mask):
        class_features = features[mask]
        if len(class_features) > 0:
            class_preds, _ = batch_predict(class_features)
            class_correct = sum(class_preds == label_idx)
            print(f"  {label_name}: {class_correct}/{len(class_features)} = {class_correct / len(class_features) * 100:.2f}%")

print("\nFirst 10 samples:")
for i in range(min(10, total)):
    true_str = true_labels[i] if isinstance(true_labels[i], str) else str(true_labels[i])
    pred_label = CLASS_LABELS.get(predicted_classes[i], str(predicted_classes[i]))
    confidence = predicted_probas[i][predicted_classes[i]]
    print(f"  [{i + 1}] true={true_str}, pred={pred_label}, conf={confidence:.2%}")

tags_reports = create_tags_report_from_predictions(predicted_classes, predicted_probas)
tags_report_df = export_tags_report_to_csv(tags_reports, "tags_report.csv")

print(f"\nTagsReport stats ({len(tags_reports)} records):")
print(f"  value range: [{tags_report_df['value'].min():.4f}, {tags_report_df['value'].max():.4f}]")
print(f"  mean: {tags_report_df['value'].mean():.4f}, std: {tags_report_df['value'].std():.4f}")

negative = tags_report_df[tags_report_df["value"] < -0.33].shape[0]
neutral = tags_report_df[(tags_report_df["value"] >= -0.33) & (tags_report_df["value"] <= 0.33)].shape[0]
positive = tags_report_df[tags_report_df["value"] > 0.33].shape[0]
print(f"\nEmotion distribution:")
print(f"  Negative: {negative} ({negative / total * 100:.1f}%)")
print(f"  Neutral:  {neutral} ({neutral / total * 100:.1f}%)")
print(f"  Positive: {positive} ({positive / total * 100:.1f}%)")

results_df = pd.DataFrame({
    "True_Label": true_labels,
    "Predicted_Label": [CLASS_LABELS.get(c, str(c)) for c in predicted_classes],
    "Predicted_Index": predicted_classes,
    "Confidence": [predicted_probas[i][predicted_classes[i]] for i in range(total)],
})
for i in range(len(CLASS_LABELS)):
    results_df[f"Prob_{CLASS_LABELS[i]}"] = predicted_probas[:, i]

results_df.to_csv("training_set_predictions.csv", index=False)
print("\nSaved predictions to training_set_predictions.csv")
