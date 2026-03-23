import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time
from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Union, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TagsReport:
    username: str
    timestamp: int
    value: float  # -1.0 to 1.0, represents likeness


_model = None
_scaler = None


def load_models():
    global _model, _scaler
    if _model is None or _scaler is None:
        logger.info("Loading model and scaler...")
        _model = load_model("best_model_feature_csv.keras")
        _scaler = joblib.load("scaler_fixed.pkl")
        logger.info("Model loaded.")
    return _model, _scaler


def predict_from_csv(csv_path: str, has_label: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {csv_path}, shape: {df.shape}")
    if has_label:
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        return features, labels, df
    return df.values, None, df


def batch_predict(features_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model, scaler = load_models()
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


def map_prediction_to_value(pred_class: int, pred_proba: np.ndarray) -> float:
    # Weighted: -1.0 * P(neg) + 0.0 * P(neu) + 1.0 * P(pos)
    if len(pred_proba) >= 3:
        value = -1.0 * pred_proba[0] + 0.0 * pred_proba[1] + 1.0 * pred_proba[2]
        return float(max(-1.0, min(1.0, value)))
    mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    return mapping.get(pred_class, 0.0)


def create_tags_report_from_predictions(
    predicted_classes: np.ndarray,
    predicted_probas: np.ndarray,
    username_prefix: str = "user",
    start_timestamp: Optional[int] = None,
) -> List[TagsReport]:
    if start_timestamp is None:
        start_timestamp = int(time.time())

    reports = []
    for i, (pred_class, pred_proba) in enumerate(zip(predicted_classes, predicted_probas)):
        reports.append(TagsReport(
            username=f"{username_prefix}_{i:04d}",
            timestamp=start_timestamp + i,
            value=map_prediction_to_value(pred_class, pred_proba),
        ))

    logger.info(f"Created {len(reports)} TagsReport objects")
    return reports


def export_tags_report_to_csv(tags_reports: List[TagsReport], output_file: str = "tags_report.csv") -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in tags_reports])[["username", "timestamp", "value"]]
    df.to_csv(output_file, index=False)
    logger.info(f"Exported TagsReports to {output_file}")
    return df


def calculate_accuracy(predicted_classes: np.ndarray, true_labels: np.ndarray) -> Dict:
    class_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label_to_idx = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    correct = 0
    total = len(true_labels)
    class_stats = {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in class_labels.values()}

    for i in range(total):
        true_idx = label_to_idx.get(true_labels[i], -1) if isinstance(true_labels[i], str) else int(true_labels[i])
        pred_idx = predicted_classes[i]
        true_label_name = class_labels.get(true_idx, "UNKNOWN")

        if true_label_name in class_stats:
            class_stats[true_label_name]["total"] += 1
            if true_idx == pred_idx:
                class_stats[true_label_name]["correct"] += 1

        if true_idx == pred_idx:
            correct += 1

    accuracy = correct / total * 100 if total > 0 else 0
    for stats in class_stats.values():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"] * 100

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "class_stats": class_stats,
    }


def predict_emotions(
    data_source: Union[str, np.ndarray, pd.DataFrame],
    has_label: bool = True,
    username_prefix: str = "user",
    export_csv: bool = True,
    output_file: str = "tags_report.csv",
) -> Dict:
    try:
        true_labels = None
        if isinstance(data_source, str):
            features, true_labels, _ = predict_from_csv(data_source, has_label)
        elif isinstance(data_source, pd.DataFrame):
            if has_label and "label" in data_source.columns:
                features = data_source.drop("label", axis=1).values
                true_labels = data_source["label"].values
            else:
                features = data_source.values
        elif isinstance(data_source, np.ndarray):
            features = data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        predicted_classes, predicted_probas = batch_predict(features)
        tags_reports = create_tags_report_from_predictions(predicted_classes, predicted_probas, username_prefix)

        stats = {
            "success": True,
            "total_samples": len(predicted_classes),
            "tags_reports": [asdict(r) for r in tags_reports],
            "message": f"Predicted {len(predicted_classes)} samples",
        }

        if true_labels is not None:
            accuracy_info = calculate_accuracy(predicted_classes, true_labels)
            stats["accuracy"] = accuracy_info["accuracy"]
            stats["class_stats"] = accuracy_info["class_stats"]
            stats["message"] += f", accuracy: {accuracy_info['accuracy']:.2f}%"

        if export_csv:
            df = export_tags_report_to_csv(tags_reports, output_file)
            stats["value_stats"] = {
                "min": float(df["value"].min()),
                "max": float(df["value"].max()),
                "mean": float(df["value"].mean()),
                "std": float(df["value"].std()),
            }
            stats["csv_path"] = output_file
            total = len(df)
            stats["emotion_distribution"] = {
                "negative": {"count": (n := df[df["value"] < -0.33].shape[0]), "percentage": n / total * 100},
                "neutral": {"count": (n := df[(df["value"] >= -0.33) & (df["value"] <= 0.33)].shape[0]), "percentage": n / total * 100},
                "positive": {"count": (n := df[df["value"] > 0.33].shape[0]), "percentage": n / total * 100},
            }

        logger.info(stats["message"])
        return stats

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"success": False, "error": str(e), "message": f"Prediction failed: {e}"}


def predict_from_features(features_array: np.ndarray, **kwargs) -> Dict:
    return predict_emotions(features_array, has_label=False, **kwargs)


def predict_from_dataframe(df: pd.DataFrame, has_label: bool = True, **kwargs) -> Dict:
    return predict_emotions(df, has_label=has_label, **kwargs)


def create_flask_app():
    try:
        from flask import Flask, request, jsonify
        app = Flask(__name__)
        load_models()

        @app.route("/health", methods=["GET"])
        def health_check():
            return jsonify({"status": "healthy", "model_loaded": _model is not None, "scaler_loaded": _scaler is not None})

        @app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.json
                if not data:
                    return jsonify({"success": False, "message": "Empty request"}), 400

                if "features" in data:
                    result = predict_from_features(np.array(data["features"]))
                elif "csv_path" in data:
                    result = predict_emotions(
                        data["csv_path"],
                        has_label=data.get("has_label", True),
                        username_prefix=data.get("username_prefix", "user"),
                        export_csv=data.get("export_csv", True),
                        output_file=data.get("output_file", "tags_report.csv"),
                    )
                else:
                    return jsonify({"success": False, "message": "Provide features or csv_path"}), 400

                return jsonify(result)
            except Exception as e:
                logger.error(f"API prediction failed: {e}")
                return jsonify({"success": False, "message": str(e)}), 500

        @app.route("/batch_predict", methods=["POST"])
        def batch_predict_endpoint():
            try:
                if "file" not in request.files or request.files["file"].filename == "":
                    return jsonify({"success": False, "message": "No file uploaded"}), 400

                file = request.files["file"]
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file.filename)
                file.save(file_path)

                result = predict_emotions(
                    file_path,
                    has_label=request.form.get("has_label", "true").lower() == "true",
                    username_prefix=request.form.get("username_prefix", "user"),
                    export_csv=request.form.get("export_csv", "true").lower() == "true",
                    output_file=request.form.get("output_file", "tags_report.csv"),
                )

                if os.path.exists(file_path):
                    os.remove(file_path)

                return jsonify(result)
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                return jsonify({"success": False, "message": str(e)}), 500

        return app

    except ImportError:
        logger.warning("Flask not installed, cannot create API server")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EEG emotion prediction")
    parser.add_argument("--csv", type=str, help="CSV file path")
    parser.add_argument("--no-label", action="store_true", help="Data has no label column")
    parser.add_argument("--output", type=str, default="tags_report.csv", help="Output CSV filename")
    parser.add_argument("--prefix", type=str, default="user", help="Username prefix")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=5000, help="API server port")

    args = parser.parse_args()

    if args.api:
        app = create_flask_app()
        if app:
            logger.info(f"Starting API server on port {args.port}")
            app.run(host="0.0.0.0", port=args.port, debug=False)
        else:
            logger.error("Cannot start API server. Install Flask: pip install flask")
    elif args.csv:
        result = predict_emotions(args.csv, has_label=not args.no_label, username_prefix=args.prefix, output_file=args.output)
        if result["success"]:
            print(f"Success: {result['message']}")
            if "csv_path" in result:
                print(f"Saved to: {result['csv_path']}")
        else:
            print(f"Failed: {result.get('message', 'Unknown error')}")
    else:
        print("Provide a CSV path or use --api to start the server")
        parser.print_help()


if __name__ == "__main__":
    main()
