# export_predictions.py
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from typing import List
import csv
import os


@dataclass
class TagsReport:
    username: str
    timestamp: int
    # Value will be a float number from -1.0 to 1.0, representing likeness.
    value: float


def map_prediction_to_value(prediction, prediction_proba=None):
    """
    将模型预测映射到-1.0到1.0的范围
    0 -> NEGATIVE -> -1.0
    1 -> NEUTRAL -> 0.0
    2 -> POSITIVE -> 1.0

    如果有预测概率，可以进行更细粒度的映射
    """
    if prediction_proba is not None:
        # 使用概率进行加权
        # 假设概率数组是 [neg_prob, neu_prob, pos_prob]
        if len(prediction_proba) >= 3:
            neg_prob = prediction_proba[0] if len(prediction_proba) > 0 else 0
            neu_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
            pos_prob = prediction_proba[2] if len(prediction_proba) > 2 else 0

            # 加权计算：-1*neg + 0*neu + 1*pos
            value = -1.0 * neg_prob + 0.0 * neu_prob + 1.0 * pos_prob
            return max(-1.0, min(1.0, value))  # 确保在-1到1之间
        elif len(prediction_proba) == 2:
            # 如果是二分类，假设是 [negative, positive]
            neg_prob = prediction_proba[0]
            pos_prob = prediction_proba[1]
            value = -1.0 * neg_prob + 1.0 * pos_prob
            return max(-1.0, min(1.0, value))

    # 如果没有概率，使用简单映射
    if prediction == 0:  # NEGATIVE
        return -1.0
    elif prediction == 1:  # NEUTRAL
        return 0.0
    elif prediction == 2:  # POSITIVE
        return 1.0
    else:
        # 如果预测值超出范围，归一化到-1到1
        return max(-1.0, min(1.0, (prediction - 1) / 1.0))  # 假设0,1,2映射到-1,0,1


def generate_username(index, prefix="user"):
    """
    生成用户名
    """
    return f"{prefix}_{index:04d}"


def generate_timestamps(n_samples, start_time=None, interval_seconds=1):
    """
    生成时间戳
    """
    if start_time is None:
        start_time = int(time.time())  # 当前时间戳

    timestamps = []
    for i in range(n_samples):
        timestamps.append(start_time + i * interval_seconds)

    return timestamps


def load_predictions_from_model(data_path="emotions.csv"):
    """
    加载模型并对数据进行预测
    """
    print("🔍 加载模型进行预测...")

    try:
        # 1. 加载模型和标准化器
        with open("emotion_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # 2. 加载数据
        data = pd.read_csv(data_path)

        # 3. 提取特征（去掉标签列）
        if "label" in data.columns:
            features = data.drop("label", axis=1).values
            true_labels = data["label"].values
        else:
            features = data.values
            true_labels = None

        # 4. 标准化
        features_scaled = scaler.transform(features)

        # 5. 预测
        predictions = model.predict(features_scaled)

        # 6. 获取预测概率（如果模型支持）
        if hasattr(model, "predict_proba"):
            prediction_probas = model.predict_proba(features_scaled)
        else:
            prediction_probas = None

        print(f"✅ 预测完成: {len(predictions)} 个样本")

        return predictions, prediction_probas, true_labels

    except Exception as e:
        print(f"❌ 加载预测失败: {e}")
        return None, None, None


def create_tags_reports(
    predictions, prediction_probas=None, username_prefix="user", start_timestamp=None
):
    """
    创建TagsReport对象列表
    """
    n_samples = len(predictions)

    # 生成用户名
    usernames = [generate_username(i, username_prefix) for i in range(n_samples)]

    # 生成时间戳
    timestamps = generate_timestamps(n_samples, start_timestamp)

    # 创建TagsReport列表
    tags_reports = []

    for i in range(n_samples):
        # 获取预测值
        pred = predictions[i]

        # 获取概率（如果有）
        proba = prediction_probas[i] if prediction_probas is not None else None

        # 映射到-1.0到1.0的范围
        value = map_prediction_to_value(pred, proba)

        # 创建TagsReport对象
        report = TagsReport(
            username=usernames[i],
            timestamp=timestamps[i],
            value=float(value),  # 确保是float类型
        )

        tags_reports.append(report)

    return tags_reports


def export_to_csv(tags_reports, output_file="model_predictions.csv"):
    """
    将TagsReport列表导出为CSV文件
    """
    print(f"📤 导出预测结果到 {output_file}...")

    # 转换为字典列表
    data = [asdict(report) for report in tags_reports]

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 确保列顺序
    df = df[["username", "timestamp", "value"]]

    # 保存为CSV
    df.to_csv(output_file, index=False)

    print(f"✅ 已导出 {len(tags_reports)} 条记录到 {output_file}")

    # 显示前几行
    print(f"\n📄 前5行数据:")
    print(df.head())

    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"value范围: [{df['value'].min():.4f}, {df['value'].max():.4f}]")
    print(f"value均值: {df['value'].mean():.4f}")
    print(f"value标准差: {df['value'].std():.4f}")

    # 分类统计
    print(f"\n🎭 情绪分布:")
    negative = df[df["value"] < -0.33].shape[0]
    neutral = df[(df["value"] >= -0.33) & (df["value"] <= 0.33)].shape[0]
    positive = df[df["value"] > 0.33].shape[0]

    total = len(df)
    print(f"负面 (value < -0.33): {negative} ({negative / total * 100:.1f}%)")
    print(f"中性 (-0.33 ≤ value ≤ 0.33): {neutral} ({neutral / total * 100:.1f}%)")
    print(f"正面 (value > 0.33): {positive} ({positive / total * 100:.1f}%)")

    return df


def export_detailed_report(
    tags_reports, predictions, true_labels=None, output_file="detailed_predictions.csv"
):
    """
    导出更详细的报告，包括原始预测值和真实标签（如果有）
    """
    print(f"\n📋 生成详细报告 {output_file}...")

    data = []
    for i, report in enumerate(tags_reports):
        row = asdict(report)
        row["prediction"] = int(predictions[i])

        # 添加预测标签
        if predictions[i] == 0:
            row["prediction_label"] = "NEGATIVE"
        elif predictions[i] == 1:
            row["prediction_label"] = "NEUTRAL"
        else:
            row["prediction_label"] = "POSITIVE"

        # 添加真实标签（如果有）
        if true_labels is not None and i < len(true_labels):
            row["true_label"] = true_labels[i]

            # 计算准确率
            if predictions[i] == 0 and true_labels[i] == "NEGATIVE":
                row["correct"] = True
            elif predictions[i] == 1 and true_labels[i] == "NEUTRAL":
                row["correct"] = True
            elif predictions[i] == 2 and true_labels[i] == "POSITIVE":
                row["correct"] = True
            else:
                row["correct"] = False

        data.append(row)

    df = pd.DataFrame(data)

    # 保存
    df.to_csv(output_file, index=False)
    print(f"✅ 详细报告已保存: {output_file}")

    # 如果有真实标签，显示准确率
    if true_labels is not None and "correct" in df.columns:
        accuracy = df["correct"].mean()
        print(f"📈 预测准确率: {accuracy:.2%}")

    return df


def main():
    """
    主函数：导出模型预测结果
    """
    print("=" * 50)
    print("📊 模型预测结果导出工具")
    print("=" * 50)

    # 1. 检查模型文件
    if not os.path.exists("emotion_model.pkl"):
        print("❌ 未找到模型文件 emotion_model.pkl")
        print("请先运行 train_model.py 训练模型")
        return

    # 2. 检查数据文件
    data_files = ["test_data.csv", "emotions_train.csv", "simple_test.csv"]
    data_file = None

    for file in data_files:
        if os.path.exists(file):
            data_file = file
            print(f"✅ 找到数据文件: {file}")
            break

    if data_file is None:
        print("❌ 未找到任何数据文件")
        print("请先创建测试数据或训练数据")
        return

    # 3. 加载并预测
    predictions, prediction_probas, true_labels = load_predictions_from_model(data_file)

    if predictions is None:
        return

    # 4. 创建TagsReport对象
    print("\n🎯 创建TagsReport对象...")
    tags_reports = create_tags_reports(predictions, prediction_probas)

    # 5. 导出为标准格式CSV
    print("\n" + "=" * 50)
    standard_df = export_to_csv(tags_reports, "model_predictions.csv")

    # 6. 导出详细报告（可选）
    print("\n" + "=" * 50)
    detailed_df = export_detailed_report(
        tags_reports, predictions, true_labels, "detailed_predictions.csv"
    )

    # 7. 创建示例数据（用于测试格式）
    print("\n" + "=" * 50)
    print("🧪 创建示例数据验证格式...")
    create_sample_format()

    print("\n" + "=" * 50)
    print("🎉 导出完成!")
    print(f"生成的文件:")
    print(f"  1. model_predictions.csv - 标准格式的预测结果")
    print(f"  2. detailed_predictions.csv - 包含详细信息的预测结果")


def create_sample_format():
    """
    创建一个示例文件展示格式
    """
    sample_data = [
        TagsReport(username="user_0001", timestamp=1672531200, value=-0.8),
        TagsReport(username="user_0002", timestamp=1672531201, value=0.0),
        TagsReport(username="user_0003", timestamp=1672531202, value=0.9),
        TagsReport(username="user_0004", timestamp=1672531203, value=-0.2),
        TagsReport(username="user_0005", timestamp=1672531204, value=1.0),
    ]

    df = pd.DataFrame([asdict(r) for r in sample_data])
    df.to_csv("sample_format.csv", index=False)

    print("✅ 示例格式文件已创建: sample_format.csv")
    print("\n📄 示例内容:")
    print(df.to_string(index=False))

    print("\n📋 格式说明:")
    print("  - username: 字符串，用户标识")
    print("  - timestamp: 整数，Unix时间戳（秒）")
    print("  - value: 浮点数，范围[-1.0, 1.0]，表示喜好程度")
    print("    * -1.0 表示非常不喜欢 (NEGATIVE)")
    print("    * 0.0  表示中性 (NEUTRAL)")
    print("    * 1.0  表示非常喜欢 (POSITIVE)")


def batch_export(data_files, output_dir="predictions"):
    """
    批量导出多个数据文件的预测结果
    """
    print("🚀 批量导出预测结果...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n处理文件: {data_file}")

            # 加载并预测
            predictions, prediction_probas, true_labels = load_predictions_from_model(
                data_file
            )

            if predictions is not None:
                # 创建TagsReport对象
                tags_reports = create_tags_reports(predictions, prediction_probas)

                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(data_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}_predictions.csv")

                # 导出
                export_to_csv(tags_reports, output_file)

    print(f"\n✅ 批量导出完成! 结果保存在 {output_dir}/ 目录")


if __name__ == "__main__":
    # 运行主函数
    main()

    # 如果需要批量处理，取消注释下面的代码
    # batch_export(['test_data.csv', 'emotions_train.csv'])
