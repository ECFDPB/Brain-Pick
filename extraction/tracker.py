import cv2
import time
import csv
import json
import numpy as np
import requests
from datetime import datetime
from GazeTracking.gaze_tracking import GazeTracking
from .attention_mapper import AttentionMapper

# API全局配置，按需修改
API_TIMEOUT = 2  # API请求超时时间（秒）

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

        # 存储解析后的Element信息
        self.screen_elements = None

    def fetch_elements_from_api(self, api_url, request_method="GET", post_data=None):
        """从服务器API拉取Element列表并解析 """
        try:
            # 发送API请求
            if request_method.upper() == "GET":
                res = requests.get(api_url, timeout=API_TIMEOUT)
            else:
                res = requests.post(api_url, json=post_data, timeout=API_TIMEOUT)
            res.raise_for_status()  # 捕获HTTP状态码错误（4xx/5xx）
            api_data = res.json()  # 解析返回的JSON为list/dict

            # 校验返回数据格式
            if not isinstance(api_data, list):
                print("API返回错误：非列表格式！")
                return None
            if len(api_data) == 0:
                print("API返回错误：元素列表为空！")
                return None

            # 解析Element，计算实际范围(x_max/y_max)
            screen_elements = []
            required_attrs = ['x', 'y', 'length', 'width', 'tag']
            for elem in api_data:
                # 校验Element属性是否完整
                if not all(attr in elem for attr in required_attrs):
                    continue
                # 转换为浮点数（防止服务器返回字符串）
                x = float(elem['x'])
                y = float(elem['y'])
                length = float(elem['length'])
                width = float(elem['width'])
                tag = elem['tag'].strip()  # 分区标签
                # 计算板块右下角坐标：x_max = 左上角x + 横向长度 | y_max = 左上角y + 纵向宽度
                x_max = x + length
                y_max = y + width
                # 过滤超出屏幕范围的无效元素（0-1）
                if x < 0 or y < 0 or x_max > 1 or y_max > 1:
                    continue
                screen_elements.append({
                    'x': x, 'y': y, 'x_max': x_max, 'y_max': y_max, 'tag': tag
                })
            return screen_elements if screen_elements else None

        except requests.exceptions.Timeout:
            print(f"API请求超时（已等待{API_TIMEOUT}秒）")
        except requests.exceptions.ConnectionError:
            print("API连接失败！请检查服务器地址/网络")
        except requests.exceptions.HTTPError as e:
            print(f"API请求失败：{e}")
        except json.JSONDecodeError:
            print("API返回错误：非合法JSON格式")
        except Exception as e:
            print(f"API拉取/解析失败：{e}")
        return None

    def match_elem_tag(self, screen_x, screen_y):
        """根据注意力坐标匹配对应的Element标签"""
        if self.screen_elements is None or len(self.screen_elements) == 0:
            return np.nan
        # 遍历元素，判断坐标是否在元素矩形范围内
        for elem in self.screen_elements:
            if elem['x'] <= screen_x <= elem['x_max'] and elem['y'] <= screen_y <= elem['y_max']:
                return elem['tag']
        return np.nan

    def get_attention_position(self):
        """获取用户当前的注意力位置"""
        ret, frame = self.webcam.read()
        if not ret:
            return {
                'username': self.username,
                'timestamp': datetime.now().isoformat(),
                'attention_tag': np.nan
            }
        # 处理帧
        self.gaze.refresh(frame)
        if not self.gaze.pupils_located:  # 核心判断：瞳孔是否被识别
            return {
                'username': self.username,
                'timestamp': datetime.now().isoformat(),
                'attention_tag': np.nan
            }
        # 跳过无效的眼睛检测 (比如眨眼时)
        if self.gaze.is_blinking():
             return {
                'username': self.username,
                'timestamp': datetime.now().isoformat(),
                'attention_tag': np.nan
            }
        # 获取眼睛位置
        gaze_h = self.gaze.horizontal_ratio()
        gaze_v = self.gaze.vertical_ratio()
        if gaze_h is None or gaze_v is None:
            return {
                'username': self.username,
                'timestamp': datetime.now().isoformat(),
                'attention_tag': np.nan
            }

        # 映射到屏幕位置
        screen_x, screen_y = self.mapper.predict(gaze_h, gaze_v)

        return {
            'username': self.username,
            'timestamp': datetime.now().isoformat(),
            'attention_tag': self.match_elem_tag(screen_x, screen_y)
        }

    def run_continuous_tracking(self, collection_rate=2, output_file='raw_data.csv'):
        """连续追踪 (不限时长)"""

        api_url = "https://127.0.0.1"  # 【必须修改】为你的实际API地址
        request_method = "GET"  # 按需改为POST
        post_data = None  # POST请求传参，例：{"user_id": self.username}
        self.screen_elements = self.fetch_elements_from_api(api_url, request_method, post_data)

        # 校验Element列表是否有效
        if self.screen_elements is None:
            raise ValueError("屏幕元素配置拉取失败，程序终止！")

        interval = 1.0 / collection_rate
        last_collection_time = time.time()
        is_running = True
        csv_file = None

        try:
            csv_file = open(output_file, 'a', newline='', encoding='utf-8')
            fieldnames=['username', 'timestamp', 'attention_tag']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


            # 写入表头
            csv_file.seek(0, 2)  # 移动到文件末尾
            if csv_file.tell() == 0:
                csv_writer.writeheader()
                csv_file.flush()


            while is_running:
                current_time = time.time()

                # 检查是否到了采样时间
                if current_time - last_collection_time >= interval:
                    result = self.get_attention_position()
                    if result is not None:
                        # 判断x_axis和y_axis是否都不是NaN
                        if not np.isnan(result['attention_tag']):
                            csv_writer.writerow(result)
                            csv_file.flush()
                    last_collection_time = current_time

                # 显示实时视频并检测ESC键
                ret, frame = self.webcam.read()
                if ret:
                    self.gaze.refresh(frame)
                    frame = self.gaze.annotated_frame()
                    cv2.imshow("Eye Tracking", frame)

                    # 检测ESC键（27是ESC的ASCII码），按下则终止循环
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        is_running = False  # 标记为停止
                        break  # 退出循环

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