# check_data.py 的内容
import pandas as pd

df = pd.read_csv("test_emotions.csv")
print("=== 数据检查报告 ===")
print(f"1. 文件总列数: {df.shape[1]}")  # 应该是2549
print(f"2. 文件总行数: {df.shape[0]}")
print(f"3. 最后一列的名称是: '{df.columns[-1]}'")
print(f"4. 最后一列的前几个值是: {df.iloc[:5, -1].tolist()}")
print("\n结论：")
print(f"   - 第1-{df.shape[1] - 1}列是【特征】，是模型的【输入】。")
print(f"   - 第{df.shape[1]}列（'{df.columns[-1]}'）是【标签】，是模型的【输出】。")
print("   - 标准化器 (scaler) 只能学习【特征】列的规律，不能包含【标签】列。")
