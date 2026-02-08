import cv2
import time
import csv
import json
import numpy as np
import requests
from datetime import datetime
from GazeTracking.gaze_tracking import GazeTracking
from .attention_mapper import AttentionMapper

API_TIMEOUT = 2  # API Timeout in seconds


class AttentionTracker:
    """Real-time eye tracking"""

    def __init__(self, username, mapper_model_path="attention_mapper.pkl"):
        self.username = username
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)

        # Webcam properties for better performance
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)

        # Load the model
        self.mapper = AttentionMapper()
        if not self.mapper.load_model(mapper_model_path):
            raise ValueError("Unable to load model, please run calibration first")

        # Store cached Elements data
        self.screen_elements = None

    def fetch_elements_from_api(self, api_url, request_method="GET", post_data=None):
        """Get the list of Elements from the server"""

        try:
            # Send API request
            if request_method.upper() == "GET":
                res = requests.get(api_url, timeout=API_TIMEOUT)
            else:
                res = requests.post(api_url, json=post_data, timeout=API_TIMEOUT)
            res.raise_for_status()  # Catch HTTP errors
            api_data = res.json()  # Parse JSON as list/dict

            # Validate data format
            if not isinstance(api_data, list):
                print("API返回错误：非列表格式！")
                return None
            if len(api_data) == 0:
                print("API返回错误：元素列表为空！")
                return None

            # Parse Element as (x_max/y_max)
            screen_elements = []
            required_attrs = ["x", "y", "length", "width", "tag"]
            for elem in api_data:
                # Validate integrity
                if not all(attr in elem for attr in required_attrs):
                    continue
                # Cast to float
                x = float(elem["x"])
                y = float(elem["y"])
                length = float(elem["length"])
                width = float(elem["width"])
                tag = elem["tag"].strip()  # Partition labels
                x_max = x + length
                y_max = y + width
                # Filter out invalid elements
                if x < 0 or y < 0 or x_max > 1 or y_max > 1:
                    continue
                screen_elements.append(
                    {"x": x, "y": y, "x_max": x_max, "y_max": y_max, "tag": tag}
                )
            return screen_elements if screen_elements else None

        except requests.exceptions.Timeout:
            print("API timeout")
        except requests.exceptions.ConnectionError:
            print("API connection failed")
        except requests.exceptions.HTTPError as e:
            print(f"API request failure: {e}")
        except json.JSONDecodeError:
            print("API failure：invalid JSON")
        except Exception as e:
            print(f"API error：{e}")
        return None

    def match_elem_tag(self, screen_x, screen_y):
        """Match an Element based on coordinates"""

        if self.screen_elements is None or len(self.screen_elements) == 0:
            return np.nan
        for elem in self.screen_elements:
            if (
                elem["x"] <= screen_x <= elem["x_max"]
                and elem["y"] <= screen_y <= elem["y_max"]
            ):
                return elem["tag"]
        return np.nan

    def get_attention_position(self):
        """Get the user's current focused coordinates"""

        ret, frame = self.webcam.read()
        if not ret:
            return {
                "username": self.username,
                "timestamp": datetime.now().isoformat(),
                "attention_tag": np.nan,
            }
        self.gaze.refresh(frame)
        if not self.gaze.pupils_located:  # Check if the pupils are detected
            return {
                "username": self.username,
                "timestamp": datetime.now().isoformat(),
                "attention_tag": np.nan,
            }
        # Skip invalid checks
        if self.gaze.is_blinking():
            return {
                "username": self.username,
                "timestamp": datetime.now().isoformat(),
                "attention_tag": np.nan,
            }
        # Get eye positions
        gaze_h = self.gaze.horizontal_ratio()
        gaze_v = self.gaze.vertical_ratio()
        if gaze_h is None or gaze_v is None:
            return {
                "username": self.username,
                "timestamp": datetime.now().isoformat(),
                "attention_tag": np.nan,
            }

        # Map to screen position
        screen_x, screen_y = self.mapper.predict(gaze_h, gaze_v)

        return {
            "username": self.username,
            "timestamp": datetime.now().isoformat(),
            "attention_tag": self.match_elem_tag(screen_x, screen_y),
        }

    def run_continuous_tracking(self, collection_rate=2, output_file="raw_data.csv"):
        api_url = "https://127.0.0.1"
        request_method = "GET"
        post_data = None
        self.screen_elements = self.fetch_elements_from_api(
            api_url, request_method, post_data
        )

        # Validate list of Elements
        if self.screen_elements is None:
            raise ValueError("Failed to get Elements")

        interval = 1.0 / collection_rate
        last_collection_time = time.time()
        is_running = True
        csv_file = None

        try:
            csv_file = open(output_file, "a", newline="", encoding="utf-8")
            fieldnames = ["username", "timestamp", "attention_tag"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            csv_file.seek(0, 2)
            if csv_file.tell() == 0:
                csv_writer.writeheader()
                csv_file.flush()

            while is_running:
                current_time = time.time()

                if current_time - last_collection_time >= interval:
                    result = self.get_attention_position()
                    if result is not None:
                        tag = result["attention_tag"]
                        if not (isinstance(tag, (int, float)) and np.isnan(tag)):
                            csv_writer.writerow(result)
                            csv_file.flush()
                    last_collection_time = current_time

                ret, frame = self.webcam.read()
                if ret:
                    self.gaze.refresh(frame)
                    frame = self.gaze.annotated_frame()
                    cv2.imshow("Eye Tracking", frame)

                    # Check for Esc key
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        is_running = False
                        break

        except KeyboardInterrupt:
            print("Interrupted")

        finally:
            # Release resources
            self.webcam.release()
            cv2.destroyAllWindows()
            if csv_file is not None:
                csv_file.close()


if __name__ == "__main__":
    username = "ecfdpb"

    try:
        tracker = AttentionTracker(username)

        tracker.run_continuous_tracking(collection_rate=2)

    except Exception as e:
        print(f"Error: {e}")

