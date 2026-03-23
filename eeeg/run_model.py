import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

model = load_model("best_model_feature_csv.keras")
scaler = joblib.load("scaler_fixed.pkl")

csv_path = "emotions.csv"


def predict_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    features = df.iloc[0, :-1].values.reshape(1, -1)
    true_label = df.iloc[0, -1]
    return features, true_label


def create_dummy_data():
    return np.random.randn(1, 2548)


new_features = create_dummy_data()
print(f"Input shape: {new_features.shape}")

new_features_scaled = scaler.transform(new_features)
new_features_reshaped = new_features_scaled[:, np.newaxis, :, np.newaxis]
print(f"Reshaped input: {new_features_reshaped.shape}")

prediction_prob = model.predict(new_features_reshaped, verbose=0)
predicted_class_idx = np.argmax(prediction_prob[0])

class_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
predicted_label = class_labels[predicted_class_idx]
confidence = prediction_prob[0][predicted_class_idx]

print("\n" + "=" * 40)
print(f"Predicted emotion: {predicted_label}")
print(f"Confidence: {confidence:.2%}")
print("Class probabilities:")
for label, prob in zip(class_labels.values(), prediction_prob[0]):
    print(f"  {label}: {prob:.2%}")
print("=" * 40)


def batch_predict(features_array):
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


print("\nBatch prediction (3 random samples):")
batch_data = np.random.randn(3, 2548)
batch_classes, batch_probas = batch_predict(batch_data)
for i, (cls, prob) in enumerate(zip(batch_classes, batch_probas)):
    print(f"  Sample {i + 1}: {class_labels[cls]} ({prob[cls]:.2%})")
