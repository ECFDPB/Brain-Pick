import cv2
import numpy as np
import time
from GazeTracking.gaze_tracking import GazeTracking


class CalibrationData:
    def __init__(self):
        self.calibration_points = []

    def add_point(self, gaze_data, screen_x, screen_y):
        """Add one calibration point"""
        self.calibration_points.append(
            {
                "gaze_horizontal_ratio": gaze_data["horizontal_ratio"],
                "gaze_vertical_ratio": gaze_data["vertical_ratio"],
                "screen_x": screen_x,
                "screen_y": screen_y,
            }
        )

    def save_calibration(self, filename="calibration_data.npy"):
        """Save the calibration data"""
        np.save(filename, np.array(self.calibration_points))


class CalibrationUI:
    """UI for calibration"""

    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)
        self.calibration_data = CalibrationData()

    def draw_calibration_point(self, frame, x, y, radius=30, color=(0, 0, 255)):
        """Draw hint on the UI"""
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 2)
        return frame

    def run_calibration(self, target_frames=10, sample_time=3.0, wait_time=1.0):
        # Normalise coordinates
        calibration_points = [
            (0.5, 0.5, "CENTER"),
            (0.2, 0.2, "TOP-LEFT"),
            (0.8, 0.2, "TOP-RIGHT"),
            (0.2, 0.8, "BOTTOM-LEFT"),
            (0.8, 0.8, "BOTTOM-RIGHT"),
        ]

        window_name = "Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.screen_width, self.screen_height)

        for norm_x, norm_y, label in calibration_points:
            # Calculate pixel coordinates
            pixel_x = norm_x * self.screen_width
            pixel_y = norm_y * self.screen_height

            preparation_start = time.time()
            while time.time() - preparation_start < sample_time:
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                self.gaze.refresh(frame)

                frame_display = cv2.flip(frame, 1)
                frame_display = self.draw_calibration_point(
                    frame_display, pixel_x, pixel_y, radius=50, color=(255, 0, 0)
                )

                elapsed = time.time() - preparation_start
                remaining = sample_time - elapsed

                cv2.putText(
                    frame_display,
                    f"Preparing: {label}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame_display,
                    f"Time: {remaining:.1f}s",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                cv2.imshow("Calibration", frame_display)

                if cv2.waitKey(1) == 27:
                    self.webcam.release()
                    cv2.destroyAllWindows()
                    return False

            """Data collection"""
            collected_frames = 0

            while collected_frames < target_frames:
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                self.gaze.refresh(frame)

                if self.gaze.pupils_located:
                    gaze_data = {
                        "horizontal_ratio": self.gaze.horizontal_ratio(),
                        "vertical_ratio": self.gaze.vertical_ratio(),
                    }
                    self.calibration_data.add_point(gaze_data, norm_x, norm_y)
                    collected_frames += 1

                frame_display = cv2.flip(frame, 1)
                frame_display = self.draw_calibration_point(
                    frame_display, pixel_x, pixel_y, radius=50, color=(0, 255, 0)
                )

                cv2.putText(
                    frame_display,
                    f"Collecting: {label}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_display,
                    f"Progress: {collected_frames}/{target_frames}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Calibration", frame_display)

                if cv2.waitKey(1) == 27:
                    self.webcam.release()
                    cv2.destroyAllWindows()
                    return False

            print(f"{label} Done, next...")
            wait_start = time.time()

            while time.time() - wait_start < wait_time:
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                self.gaze.refresh(frame)

                frame_display = cv2.flip(frame, 1)
                cv2.putText(
                    frame_display,
                    "Next point loading...",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame_display,
                    f"{label} Completed",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Calibration", frame_display)

                if cv2.waitKey(1) == 27:
                    self.webcam.release()
                    cv2.destroyAllWindows()
                    return False

        cv2.destroyAllWindows()
        self.calibration_data.save_calibration()
        self.webcam.release()
        return True

    def release(self):
        if self.webcam:
            self.webcam.release()


if __name__ == "__main__":
    # Screen properties
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720

    calibration = CalibrationUI(SCREEN_WIDTH, SCREEN_HEIGHT)
    use_fullscreen = False
    calibration.run_calibration(target_frames=10, sample_time=3.0, wait_time=1.0)
    calibration.release()

