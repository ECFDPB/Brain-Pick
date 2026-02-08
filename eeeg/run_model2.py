import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# 1. 加载训练好的模型和标准化器
print("正在加载模型和标准化器...")
model = load_model("best_model_feature_csv.keras")
scaler = joblib.load("scaler_fixed.pkl")
print("加载完成！")


def predict_from_csv(csv_path, has_label=True):
    """从CSV文件读取数据进行预测"""
    df_new = pd.read_csv(csv_path)

    if has_label:
        # 如果有标签列，分离特征和标签
        features = df_new.iloc[:, :-1].values  # 所有行，除了最后一列
        labels = df_new.iloc[:, -1].values  # 最后一列是标签
        return features, labels
    else:
        # 如果没有标签列，直接返回特征
        features = df_new.values
        return features, None


def batch_predict(features_array):
    """对多行数据进行批量预测"""
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas


# 分类标签映射
class_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# 2. 使用训练集进行预测
csv_path = "emotions 2.csv"  # 你的训练集
print(f"开始处理数据文件: {csv_path}")

# 读取数据
features, true_labels = predict_from_csv(csv_path, has_label=True)
print(f"数据特征形状: {features.shape}")
print(f"真实标签数量: {len(true_labels)}")

# 3. 批量预测
print("开始批量预测...")
predicted_classes, predicted_probas = batch_predict(features)

# 4. 计算准确率
correct = 0
total = len(true_labels)

# 将字符串标签映射为数字（如果标签是字符串）
label_to_idx = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

# 计算准确率
for i in range(total):
    # 如果标签是字符串，转换为数字
    if isinstance(true_labels[i], str):
        true_idx = label_to_idx.get(true_labels[i], -1)
    else:
        true_idx = int(true_labels[i])

    predicted_idx = predicted_classes[i]

    if true_idx == predicted_idx:
        correct += 1

accuracy = correct / total * 100

# 5. 输出结果
print("\n" + "=" * 50)
print("训练集预测结果：")
print(f"  总样本数: {total}")
print(f"  正确预测数: {correct}")
print(f"  准确率: {accuracy:.2f}%")
print("\n  分类报告:")

# 计算每个类别的准确率
for label_name, label_idx in label_to_idx.items():
    mask = [
        True
        if (isinstance(l, str) and label_to_idx.get(l, -1) == label_idx)
        else (l == label_idx)
        for l in true_labels
    ]

    if any(mask):
        class_features = features[mask]
        class_true_labels = np.array(true_labels)[mask]

        # 预测这个类别
        if len(class_features) > 0:
            class_preds, _ = batch_predict(class_features)
            class_correct = sum(class_preds == label_idx)
            class_total = len(class_features)
            class_acc = class_correct / class_total * 100

            print(f"    {label_name}: {class_correct}/{class_total} = {class_acc:.2f}%")

# 6. 显示一些样本的详细预测结果
print("\n  前10个样本的预测详情:")
for i in range(min(10, total)):
    if isinstance(true_labels[i], str):
        true_label_str = true_labels[i]
        true_idx = label_to_idx.get(true_labels[i], -1)
    else:
        true_idx = int(true_labels[i])
        true_label_str = [k for k, v in class_labels.items() if v == true_labels[i]]
        true_label_str = true_label_str[0] if true_label_str else str(true_labels[i])

    pred_idx = predicted_classes[i]
    pred_label = class_labels.get(pred_idx, f"未知({pred_idx})")
    confidence = predicted_probas[i][pred_idx]

    print(
        f"    样本{i + 1}: 真实={true_label_str}, 预测={pred_label}, 置信度={confidence:.2%}"
    )

print("=" * 50)

# 7. 保存预测结果到CSV（可选）
results_df = pd.DataFrame(
    {
        "True_Label": true_labels,
        "Predicted_Label": [
            class_labels.get(c, f"未知({c})") for c in predicted_classes
        ],
        "Predicted_Index": predicted_classes,
        "Confidence": [predicted_probas[i][predicted_classes[i]] for i in range(total)],
    }
)

# 添加每个类别的概率
for i in range(len(class_labels)):
    results_df[f"Prob_{class_labels[i]}"] = predicted_probas[:, i]

results_df.to_csv("training_set_predictions.csv", index=False)
print("预测结果已保存到 'training_set_predictions.csv'")
