import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from EEGModels import EEGNet
import matplotlib
import matplotlib.pyplot as plt
import joblib

matplotlib.use("Agg")

df = pd.read_csv("emotions.csv")
print(f"Dataset shape: {df.shape}")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]
print(f"Features: {X.shape}, Labels: {y_categorical.shape}, Classes: {num_classes}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Reshape to (samples, 1, features, 1) for EEGNet
X_train_reshaped = X_train[:, np.newaxis, :, np.newaxis]
X_test_reshaped = X_test[:, np.newaxis, :, np.newaxis]
print(f"Reshaped train shape: {X_train_reshaped.shape}")

model = EEGNet(
    nb_classes=num_classes,
    Chans=1,
    Samples=X_train_reshaped.shape[2],
    dropoutRate=0.5,
    kernLength=32,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model_feature_csv.keras", save_best_only=True),
]

history = model.fit(
    X_train_reshaped,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_history_csv.png")

model.save("best_model_feature_csv.keras")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
