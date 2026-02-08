import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


class AttentionMapper:
    """Map eye coordinates to screen positions"""

    def __init__(self):
        self.gaze_to_screen_x = None
        self.gaze_to_screen_y = None

    def train(self, calibration_data):
        """Use regression model to learn the mapping relationship"""
        if len(calibration_data) < 5:
            raise ValueError("At least 5 points are needed")

        gaze_horizontal = np.array(
            [p["gaze_horizontal_ratio"] for p in calibration_data]
        )
        gaze_vertical = np.array([p["gaze_vertical_ratio"] for p in calibration_data])
        screen_x = np.array([p["screen_x"] for p in calibration_data])
        screen_y = np.array([p["screen_y"] for p in calibration_data])

        # Combine traits
        X = np.column_stack([gaze_horizontal, gaze_vertical])

        self.gaze_to_screen_x = LinearRegression()
        self.gaze_to_screen_x.fit(X, screen_x)

        self.gaze_to_screen_y = LinearRegression()
        self.gaze_to_screen_y.fit(X, screen_y)

        print("Training complete")

    def predict(self, gaze_horizontal_ratio, gaze_vertical_ratio):
        if self.gaze_to_screen_x is None or self.gaze_to_screen_y is None:
            raise ValueError("Please train a model first")

        X = np.array([[gaze_horizontal_ratio, gaze_vertical_ratio]])

        screen_x = self.gaze_to_screen_x.predict(X)[0]
        screen_y = self.gaze_to_screen_y.predict(X)[0]

        # Clip within normalised coordinates
        screen_x = np.clip(screen_x, 0, 1)
        screen_y = np.clip(screen_y, 0, 1)

        return screen_x, screen_y

    def save_model(self, filename="attention_mapper.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "gaze_to_screen_x": self.gaze_to_screen_x,
                    "gaze_to_screen_y": self.gaze_to_screen_y,
                },
                f,
            )
        print(f"Saved to {filename}")

    def load_model(self, filename="attention_mapper.pkl"):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.gaze_to_screen_x = data["gaze_to_screen_x"]
                self.gaze_to_screen_y = data["gaze_to_screen_y"]
            print("Loaded")
            return True
        except:
            return False
