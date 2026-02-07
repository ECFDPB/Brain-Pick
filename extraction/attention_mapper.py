import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


class AttentionMapper:
    """将眼睛位置映射到屏幕注意力位置"""

    def __init__(self):
        self.gaze_to_screen_x = None
        self.gaze_to_screen_y = None

    def train(self, calibration_data):
        """使用校准数据训练映射模型

        使用多项式回归来学习眼睛位置到屏幕位置的映射关系
        """
        if len(calibration_data) < 5:
            raise ValueError("需要至少5个校准点来训练模型")

        # 提取数据
        gaze_horizontal = np.array([p['gaze_horizontal_ratio'] for p in calibration_data])
        gaze_vertical = np.array([p['gaze_vertical_ratio'] for p in calibration_data])
        screen_x = np.array([p['screen_x'] for p in calibration_data])
        screen_y = np.array([p['screen_y'] for p in calibration_data])

        # 组合眼睛特征 (水平和竖直比例)
        X = np.column_stack([gaze_horizontal, gaze_vertical])

        # 训练水平位置模型
        self.gaze_to_screen_x = LinearRegression()
        self.gaze_to_screen_x.fit(X, screen_x)

        # 训练竖直位置模型
        self.gaze_to_screen_y = LinearRegression()
        self.gaze_to_screen_y.fit(X, screen_y)

        print("注意力映射模型训练完成！")

    def predict(self, gaze_horizontal_ratio, gaze_vertical_ratio):
        """预测屏幕注意力位置

        Args:
            gaze_horizontal_ratio: 眼睛的水平比例 (0-1)
            gaze_vertical_ratio: 眼睛的竖直比例 (0-1)

        Returns:
            (screen_x, screen_y): 屏幕注意力位置 (0-1)
        """
        if self.gaze_to_screen_x is None or self.gaze_to_screen_y is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        X = np.array([[gaze_horizontal_ratio, gaze_vertical_ratio]])

        screen_x = self.gaze_to_screen_x.predict(X)[0]
        screen_y = self.gaze_to_screen_y.predict(X)[0]

        # 限制在 0-1 范围内
        screen_x = np.clip(screen_x, 0, 1)
        screen_y = np.clip(screen_y, 0, 1)

        return screen_x, screen_y

    def save_model(self, filename='attention_mapper.pkl'):
        """保存模型"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'gaze_to_screen_x': self.gaze_to_screen_x,
                'gaze_to_screen_y': self.gaze_to_screen_y
            }, f)
        print(f"模型已保存到 {filename}")

    def load_model(self, filename='attention_mapper.pkl'):
        """加载模型"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.gaze_to_screen_x = data['gaze_to_screen_x']
                self.gaze_to_screen_y = data['gaze_to_screen_y']
            print(f"模型已加载")
            return True
        except:
            return False