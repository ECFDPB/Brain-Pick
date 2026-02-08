import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib  # 或 import joblib

# 1. 加载训练好的模型和标准化器
print("正在加载模型和标准化器...")
model = load_model("best_model_feature_csv.keras")  # 修改为你的模型路径
scaler = joblib.load("scaler_fixed.pkl")  # 修改为你的标准化器路径
print("加载完成！")

csv_path = "emotions.csv"


# 2. 准备一条新的数据
# 方式A: 从新的CSV文件读取（最常用）
def predict_from_csv(csv_path):
    """从CSV文件读取一行数据进行预测"""
    df_new = pd.read_csv(csv_path)
    # 假设CSV格式与训练数据完全相同（特征列+标签列）
    # 这里我们取第一行（不含标签列）进行演示
    new_features = df_new.iloc[0, :-1].values.reshape(1, -1)  # 形状 (1, 2549)
    true_label = df_new.iloc[0, -1]  # 真实的标签（如果有）
    return new_features, true_label


# 方式B: 手动创建一个模拟数据（用于快速测试）
def create_dummy_data():
    """创建一个形状正确的随机数据向量"""
    # 你的模型期望2549个特征，这里用随机数模拟
    dummy_features = np.random.randn(1, 2548)  # 1行，2549列
    return dummy_features


# 选择一种方式获取新数据
# new_features, true_label = predict_from_csv('new_eeg_data.csv') # 使用方式A
new_features = create_dummy_data()  # 使用方式B（测试用）
print(f"新数据特征形状: {new_features.shape}")

# 3. 关键：使用相同的标准化器对新数据进行变换
new_features_scaled = scaler.transform(new_features)
print("数据标准化完成。")

# 4. 将数据重塑为模型期望的输入形状 (样本数, 1, 特征长度, 1)
# 这与训练时的 reshape 操作必须完全一致！
new_features_reshaped = new_features_scaled[:, np.newaxis, :, np.newaxis]
print(f"重塑后输入形状: {new_features_reshaped.shape}")

# 5. 进行预测
prediction_prob = model.predict(new_features_reshaped, verbose=0)  # 得到概率
predicted_class_idx = np.argmax(prediction_prob[0])  # 取概率最高的类别索引

# 6. 解读结果
class_labels = {0: "消极 (Negative)", 1: "中立 (Neutral)", 2: "积极 (Positive)"}
predicted_label = class_labels[predicted_class_idx]
confidence = prediction_prob[0][predicted_class_idx]

print("\n" + "=" * 50)
print("模型预测结果：")
print(f"  预测情绪: {predicted_label}")
print(f"  置信度: {confidence:.2%}")
print(f"  各类别详细概率:")
for idx, (label, prob) in enumerate(zip(class_labels.values(), prediction_prob[0])):
    print(f"    {label}: {prob:.2%}")
# 如果使用方式A且有真实标签，可以对比
# print(f"  真实标签: {true_label}")
print("=" * 50)


# （可选）7. 批量预测示例
def batch_predict(features_array):
    """对多行数据（一个数组）进行批量预测"""
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


# 示例：批量预测3条随机数据
print("\n批量预测示例（3条随机数据）:")
batch_data = np.random.randn(3, 2548)
batch_classes, batch_probas = batch_predict(batch_data)
for i, (cls, prob) in enumerate(zip(batch_classes, batch_probas)):
    print(f"  样本{i + 1}: 预测 -> {class_labels[cls]}, 置信度:{prob[cls]:.2%}")
