# process_training_predictions.py
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from emotion_to_tagsreport import EmotionToTagsReportConverter, TagsReport, Tag


class TrainingPredictionsProcessor:
    """处理训练集预测结果的处理器"""

    def __init__(self, csv_path: str, username: str = "EFBPE"):
        """
        初始化处理器

        参数:
        csv_path: CSV文件路径
        username: 用户名
        """
        self.csv_path = csv_path
        self.converter = EmotionToTagsReportConverter(username=username)
        self.data = None

    def load_data(self):
        """加载CSV数据"""
        print(f"正在加载数据: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        print(f"数据列: {list(self.data.columns)}")

    def convert_to_emotion_result(self, row) -> Dict[str, Any]:
        """
        将CSV行转换为情绪结果字典

        参数:
        row: DataFrame行

        返回:
        情绪结果字典
        """
        # 确保情绪标签是大写的
        predicted_label = str(row["Predicted_Label"]).upper()

        emotion_result = {
            "emotion": predicted_label,
            "confidence": float(row["Confidence"]),
            "probabilities": {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            },
        }

        return emotion_result

    def process_all_predictions(self) -> List[TagsReport]:
        """
        处理所有预测结果

        返回:
        TagsReport列表
        """
        if self.data is None:
            self.load_data()

        emotion_results = []

        for idx, row in self.data.iterrows():
            emotion_result = self.convert_to_emotion_result(row)
            emotion_results.append(emotion_result)

        # 批量转换
        base_timestamp = int(time.time())
        tags_reports = []

        for i, result in enumerate(emotion_results):
            # 为每个结果分配时间戳（递增1秒）
            timestamp = base_timestamp + i
            tags_report = self.converter.convert_with_probabilities(result, timestamp)
            tags_reports.append(tags_report)

        return tags_reports

    def process_sample(self, sample_size: int = 10) -> List[TagsReport]:
        """
        处理样本数据

        参数:
        sample_size: 样本大小

        返回:
        TagsReport列表
        """
        if self.data is None:
            self.load_data()

        # 随机选择样本
        sample_data = self.data.sample(
            n=min(sample_size, len(self.data)), random_state=42
        )

        tags_reports = []
        base_timestamp = int(time.time())

        for i, (idx, row) in enumerate(sample_data.iterrows()):
            emotion_result = self.convert_to_emotion_result(row)

            print(f"\n样本 {i + 1}:")
            print(f"  真实标签: {row['True_Label']}")
            print(f"  预测标签: {row['Predicted_Label']}")
            print(f"  置信度: {row['Confidence']:.4f}")
            print(
                f"  概率分布: NEGATIVE={row['Prob_NEGATIVE']:.4f}, "
                f"NEUTRAL={row['Prob_NEUTRAL']:.4f}, "
                f"POSITIVE={row['Prob_POSITIVE']:.4f}"
            )

            timestamp = base_timestamp + i
            tags_report = self.converter.convert_with_probabilities(
                emotion_result, timestamp
            )
            tags_reports.append(tags_report)

            print(f"  转换结果:")
            print(f"    likeness: {tags_report.value:.4f}")
            print(f"    标签数: {len(tags_report.topic)}")
            print(f"    主要标签: {[tag.name for tag in tags_report.topic[:3]]}")

        return tags_reports

    def save_to_json(self, tags_reports: List[TagsReport], output_path: str):
        """
        将TagsReports保存为JSON文件

        参数:
        tags_reports: TagsReport列表
        output_path: 输出文件路径
        """
        reports_dict = []

        for report in tags_reports:
            report_dict = self.converter.to_dict(report)
            reports_dict.append(report_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reports_dict, f, indent=2, ensure_ascii=False)

        print(f"\n已保存 {len(tags_reports)} 个报告到 {output_path}")

    def analyze_results(self, tags_reports: List[TagsReport]):
        """
        分析转换结果

        参数:
        tags_reports: TagsReport列表
        """
        print("\n" + "=" * 50)
        print("转换结果分析")
        print("=" * 50)

        # 计算统计信息
        likeness_values = [report.value for report in tags_reports]
        tag_counts = [len(report.topic) for report in tags_reports]

        print(f"总报告数: {len(tags_reports)}")
        print(
            f"likeness值范围: [{min(likeness_values):.4f}, {max(likeness_values):.4f}]"
        )
        print(f"平均likeness值: {np.mean(likeness_values):.4f}")
        print(f"likeness值标准差: {np.std(likeness_values):.4f}")
        print(f"平均标签数: {np.mean(tag_counts):.2f}")
        print(f"标签数范围: [{min(tag_counts)}, {max(tag_counts)}]")

        # 分布分析
        print(f"\nlikeness值分布:")
        print(f"  高度负向 (<= -0.8): {sum(1 for v in likeness_values if v <= -0.8)}")
        print(
            f"  中度负向 (-0.8 < v <= -0.3): {sum(1 for v in likeness_values if -0.8 < v <= -0.3)}"
        )
        print(
            f"  轻度负向 (-0.3 < v < 0): {sum(1 for v in likeness_values if -0.3 < v < 0)}"
        )
        print(f"  中立 (v = 0): {sum(1 for v in likeness_values if v == 0)}")
        print(
            f"  轻度正向 (0 < v < 0.3): {sum(1 for v in likeness_values if 0 < v < 0.3)}"
        )
        print(
            f"  中度正向 (0.3 <= v < 0.8): {sum(1 for v in likeness_values if 0.3 <= v < 0.8)}"
        )
        print(f"  高度正向 (>= 0.8): {sum(1 for v in likeness_values if v >= 0.8)}")

        # 分析常见标签
        tag_frequency = {}
        for report in tags_reports:
            for tag in report.topic:
                tag_frequency[tag.name] = tag_frequency.get(tag.name, 0) + 1

        print(f"\n最常见的10个标签:")
        sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
        for tag_name, count in sorted_tags[:10]:
            percentage = (count / len(tags_reports)) * 100
            print(f"  {tag_name}: {count}次 ({percentage:.1f}%)")

    def compare_with_true_labels(self):
        """
        将转换结果与真实标签比较

        注意：这个方法需要CSV中有True_Label列
        """
        if self.data is None:
            self.load_data()

        print("\n" + "=" * 50)
        print("与真实标签比较")
        print("=" * 50)

        correct_mapping = 0
        incorrect_mapping = 0

        for idx, row in self.data.iterrows():
            true_label = str(row["True_Label"]).upper()
            predicted_label = str(row["Predicted_Label"]).upper()

            # 计算likeness值
            probs = {
                "NEGATIVE": float(row["Prob_NEGATIVE"]),
                "NEUTRAL": float(row["Prob_NEUTRAL"]),
                "POSITIVE": float(row["Prob_POSITIVE"]),
            }

            # 计算加权likeness值
            likeness = 0.0
            for emotion_name, prob in probs.items():
                base_value = self.converter.EMOTION_TO_LIKENESS.get(
                    emotion_name.upper(), 0.0
                )
                likeness += base_value * prob

            likeness = max(-1.0, min(1.0, likeness))

            # 根据likeness判断映射的情绪类别
            if likeness <= -0.5:
                mapped_label = "NEGATIVE"
            elif likeness >= 0.5:
                mapped_label = "POSITIVE"
            else:
                mapped_label = "NEUTRAL"

            # 检查映射是否正确
            if mapped_label == true_label:
                correct_mapping += 1
            else:
                incorrect_mapping += 1

        total = len(self.data)
        accuracy = (correct_mapping / total) * 100 if total > 0 else 0

        print(f"总样本数: {total}")
        print(f"正确映射: {correct_mapping} ({accuracy:.2f}%)")
        print(f"错误映射: {incorrect_mapping}")

        return accuracy


def main():
    """主函数"""
    print("训练集预测结果处理")
    print("=" * 50)

    # 初始化处理器
    processor = TrainingPredictionsProcessor(
        csv_path="training_set_predictions.csv", username="EFBPE"
    )

    # 1. 处理样本数据（演示）
    print("\n1. 处理样本数据")
    sample_reports = processor.process_sample(sample_size=5)

    # 显示样本的JSON格式
    print(f"\n第一个样本的JSON格式:")
    print(processor.converter.to_json(sample_reports[0]))

    # 2. 处理所有数据
    print("\n\n2. 处理所有数据")
    all_reports = processor.process_all_predictions()

    # 3. 分析结果
    processor.analyze_results(all_reports)

    # 4. 与真实标签比较
    processor.compare_with_true_labels()

    # 5. 保存结果到JSON文件
    output_path = f"tags_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    processor.save_to_json(all_reports, output_path)

    # 6. 生成汇总报告
    print("\n\n3. 生成汇总报告")

    # 创建汇总数据
    summary = {
        "processing_date": datetime.now().isoformat(),
        "username": "EFBPE",
        "total_reports": len(all_reports),
        "likeness_statistics": {
            "mean": float(np.mean([r.value for r in all_reports])),
            "std": float(np.std([r.value for r in all_reports])),
            "min": float(min([r.value for r in all_reports])),
            "max": float(max([r.value for r in all_reports])),
        },
        "sample_reports": [
            processor.converter.to_dict(report) for report in all_reports[:3]
        ],
    }

    # 保存汇总报告
    summary_path = (
        f"tags_reports_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"汇总报告已保存到: {summary_path}")

    print("\n" + "=" * 50)
    print("处理完成!")


if __name__ == "__main__":
    main()
