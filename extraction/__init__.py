from .calibration import CalibrationUI
from .tracker import AttentionTracker
from .attention_mapper import AttentionMapper
import numpy as np


def run_calibration():
    width = 1280
    height = 720

    calibration = CalibrationUI(width, height)
    success = calibration.run_calibration()

    if success:
        calibration_data = np.load("calibration_data.npy", allow_pickle=True).tolist()
        mapper = AttentionMapper()
        mapper.train(calibration_data)
        mapper.save_model()


def run_tracking_continuous():
    try:
        tracker = AttentionTracker("ecfdpb")
        tracker.run_continuous_tracking(collection_rate=2)
    except Exception as e:
        print(f"Error: {e}")


def main():
    run_calibration()
    run_tracking_continuous()


if __name__ == "__main__":
    main()
