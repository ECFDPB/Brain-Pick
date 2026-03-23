import numpy as np
import pandas as pd


def create_test_data(num_samples=100, num_features=2548, csv_path="test_emotions.csv"):
    features = np.random.uniform(-1000, 1000, size=(num_samples, num_features))
    labels = np.random.choice(["NEGATIVE", "NEUTRAL", "POSITIVE"], size=num_samples)

    feature_columns = [f"feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(features, columns=feature_columns)
    df["label"] = labels
    df.to_csv(csv_path, index=False)

    print(f"Saved to: {csv_path}, shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    create_test_data(num_samples=99, num_features=2548, csv_path="test_emotions_correct.csv")
