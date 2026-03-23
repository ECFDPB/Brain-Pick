import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

model = load_model("best_model_feature_csv.keras")
scaler = joblib.load("scaler_fixed.pkl")

CLASS_LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL_TO_IDX = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


def predict_from_csv(csv_path, has_label=True):
    df = pd.read_csv(csv_path)
    if has_label:
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        return features, labels
    return df.values, None


def batch_predict(features_array):
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


csv_path = "emotions 2.csv"
print(f"Loading: {csv_path}")

features, true_labels = predict_from_csv(csv_path, has_label=True)
print(f"Features: {features.shape}, Labels: {len(true_labels)}")

predicted_classes, predicted_probas = batch_predict(features)

correct = 0
total = len(true_labels)

for i in range(total):
    true_idx = LABEL_TO_IDX.get(true_labels[i], -1) if isinstance(true_labels[i], str) else int(true_labels[i])
    if true_idx == predicted_classes[i]:
        correct += 1

accuracy = correct / total * 100

print(f"\nResults on training set:")
print(f"  Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")
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
            class_total = len(class_features)
            print(f"  {label_name}: {class_correct}/{class_total} = {class_correct / class_total * 100:.2f}%")

print("\nFirst 10 samples:")
for i in range(min(10, total)):
    true_label_str = true_labels[i] if isinstance(true_labels[i], str) else str(true_labels[i])
    pred_label = CLASS_LABELS.get(predicted_classes[i], str(predicted_classes[i]))
    confidence = predicted_probas[i][predicted_classes[i]]
    print(f"  [{i + 1}] true={true_label_str}, pred={pred_label}, conf={confidence:.2%}")

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
