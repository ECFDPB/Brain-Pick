import cv2
import time
import csv
from datetime import datetime
from GazeTracking.gaze_tracking import GazeTracking
from .attention_mapper import AttentionMapper


class AttentionTracker:
    """实时眼睛注意力追踪"""

    def __init__(self, username, mapper_model_path='attention_mapper.pkl'):
        self.username = username
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)

        # 设置摄像头属性以提高性能
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)

        # 加载注意力映射模型
        self.mapper = AttentionMapper()
        if not self.mapper.load_model(mapper_model_path):
            raise ValueError("无法加载映射模型，请先运行校准程序")

    def get_attention_position(self):
        """获取用户当前的注意力位置"""
        ret, frame = self.webcam.read()
        if not ret:
            return None
        # 处理帧
        self.gaze.refresh(frame)

        # 获取眼睛位置
        gaze_h = self.gaze.horizontal_ratio()
        gaze_v = self.gaze.vertical_ratio()

        # 跳过无效的眼睛检测 (比如眨眼时)
        if self.gaze.is_blinking():
            return {
            'username': self.username,
            'timestamp': datetime.now().isoformat(),
            'x_axis': NaN,
            'y_axis': NaN
        }

        # 映射到屏幕位置
        screen_x, screen_y = self.mapper.predict(gaze_h, gaze_v)

        return {
            'username': self.username,
            'timestamp': datetime.now().isoformat(),
            'x_axis': round(screen_x, 3),
            'y_axis': round(screen_y, 3)
        }


    def run_continuous_tracking(self, collection_rate=2, output_file='raw_data.csv'):
        """连续追踪 (不限时长)"""

        interval = 1.0 / collection_rate
        last_collection_time = time.time()
        results_count = 0

        # 初始化 csv_file 变量
        csv_file = None

        try:
            csv_file = open(output_file, 'w', newline='', encoding='utf-8')
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=['username', 'timestamp', 'x_axis', 'y_axis']
            )

            # 写入表头
            csv_writer.writeheader()
            csv_file.flush()


            while True:
                current_time = time.time()

                # 检查是否到了采样时间
                if current_time - last_collection_time >= interval:
                    result = self.get_attention_position()

                    if result is not None:
                        # 写入 CSV 文件
                        csv_writer.writerow(result)
                        csv_file.flush()
                        results_count += 1

                    last_collection_time = current_time

                # 显示实时视频
                ret, frame = self.webcam.read()
                if ret:
                    self.gaze.refresh(frame)
                    frame = self.gaze.annotated_frame()

                    cv2.imshow("Eye Tracking", frame)

                    if cv2.waitKey(1) == 27:  # ESC 退出
                        break

        except KeyboardInterrupt:
            print("\n\n追踪已停止")


        finally:

            # 关闭资源
            self.webcam.release()
            cv2.destroyAllWindows()
            if csv_file is not None:
                csv_file.close()


if __name__ == "__main__":
    # 使用示例
    username = 'ecfdpb'

    try:
        tracker = AttentionTracker(username)

        # 每秒收集2次
        tracker.run_continuous_tracking(collection_rate=2)

    except Exception as e:
        print(f"错误: {e}")