import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time
from dataclasses import dataclass, asdict
from datetime import datetime

# 定义 TagsReport 数据类
@dataclass
class TagsReport:
    username: str
    timestamp: int
    # Value will be a float number from -1.0 to 1.0, representing likeness.
    value: float

# 1. 加载训练好的模型和标准化器
print("正在加载模型和标准化器...")
model = load_model('best_model_feature_csv.keras')
scaler = joblib.load('scaler_fixed.pkl')
print("加载完成！")

def predict_from_csv(csv_path, has_label=True):
    """从CSV文件读取数据进行预测"""
    df_new = pd.read_csv(csv_path)
    
    if has_label:
        # 如果有标签列，分离特征和标签
        features = df_new.iloc[:, :-1].values  # 所有行，除了最后一列
        labels = df_new.iloc[:, -1].values  # 最后一列是标签
        return features, labels, df_new
    else:
        # 如果没有标签列，直接返回特征
        features = df_new.values
        return features, None, df_new

def batch_predict(features_array):
    """对多行数据进行批量预测"""
    features_scaled = scaler.transform(features_array)
    features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
    probas = model.predict(features_reshaped, verbose=0)
    classes = np.argmax(probas, axis=1)
    return classes, probas

def map_prediction_to_value(pred_class, pred_proba):
    """
    将模型预测映射到-1.0到1.0的范围
    使用概率加权：-1.0 * P(NEGATIVE) + 0.0 * P(NEUTRAL) + 1.0 * P(POSITIVE)
    """
    if len(pred_proba) >= 3:
        neg_prob = pred_proba[0] if len(pred_proba) > 0 else 0
        neu_prob = pred_proba[1] if len(pred_proba) > 1 else 0
        pos_prob = pred_proba[2] if len(pred_proba) > 2 else 0
        
        # 加权计算：-1*neg + 0*neu + 1*pos
        value = -1.0 * neg_prob + 0.0 * neu_prob + 1.0 * pos_prob
        return max(-1.0, min(1.0, value))  # 确保在-1到1之间
    else:
        # 如果没有概率，使用简单映射
        if pred_class == 0:  # NEGATIVE
            return -1.0
        elif pred_class == 1:  # NEUTRAL
            return 0.0
        else:  # POSITIVE
            return 1.0

def create_tags_report_from_predictions(predicted_classes, predicted_probas, 
                                       username_prefix="user", start_timestamp=None):
    """
    将预测结果转换为TagsReport格式
    """
    if start_timestamp is None:
        start_timestamp = int(time.time())
    
    tags_reports = []
    
    for i, (pred_class, pred_proba) in enumerate(zip(predicted_classes, predicted_probas)):
        # 生成用户名
        username = f"{username_prefix}_{i:04d}"
        
        # 生成时间戳（递增1秒）
        timestamp = start_timestamp + i
        
        # 计算value值
        value = map_prediction_to_value(pred_class, pred_proba)
        
        # 创建TagsReport对象
        report = TagsReport(
            username=username,
            timestamp=timestamp,
            value=float(value)
        )
        
        tags_reports.append(report)
    
    return tags_reports

def export_tags_report_to_csv(tags_reports, output_file='tags_report.csv'):
    """
    将TagsReport列表导出为CSV文件
    """
    # 转换为字典列表
    data = [asdict(report) for report in tags_reports]
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 确保列顺序
    df = df[['username', 'timestamp', 'value']]
    
    # 保存为CSV
    df.to_csv(output_file, index=False)
    
    return df

# 分类标签映射
class_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

# 2. 使用训练集进行预测
csv_path = "emotions 2.csv"  # 你的训练集
print(f"开始处理数据文件: {csv_path}")

# 读取数据
features, true_labels, original_df = predict_from_csv(csv_path, has_label=True)
print(f"数据特征形状: {features.shape}")
print(f"真实标签数量: {len(true_labels)}")

# 3. 批量预测
print("开始批量预测...")
predicted_classes, predicted_probas = batch_predict(features)

# 4. 计算准确率
correct = 0
total = len(true_labels)

# 将字符串标签映射为数字（如果标签是字符串）
label_to_idx = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

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
print("\n" + "="*50)
print("训练集预测结果：")
print(f"  总样本数: {total}")
print(f"  正确预测数: {correct}")
print(f"  准确率: {accuracy:.2f}%")
print("\n  分类报告:")

# 计算每个类别的准确率
for label_name, label_idx in label_to_idx.items():
    mask = [True if (isinstance(l, str) and label_to_idx.get(l, -1) == label_idx) 
            else (l == label_idx) for l in true_labels]
    
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
    
    print(f"    样本{i+1}: 真实={true_label_str}, 预测={pred_label}, 置信度={confidence:.2%}")

print("="*50)

# 7. 将预测结果转换为TagsReport格式并保存为CSV
print("\n正在将预测结果转换为TagsReport格式...")
tags_reports = create_tags_report_from_predictions(predicted_classes, predicted_probas)
tags_report_df = export_tags_report_to_csv(tags_reports, 'tags_report.csv')

print(f"✅ TagsReport格式的预测结果已保存到 'tags_report.csv'")
print(f"\n📊 TagsReport统计信息:")
print(f"  总记录数: {len(tags_reports)}")
print(f"  value范围: [{tags_report_df['value'].min():.4f}, {tags_report_df['value'].max():.4f}]")
print(f"  value均值: {tags_report_df['value'].mean():.4f}")
print(f"  value标准差: {tags_report_df['value'].std():.4f}")

# 分类统计
print(f"\n🎭 情绪分布:")
negative = tags_report_df[tags_report_df['value'] < -0.33].shape[0]
neutral = tags_report_df[(tags_report_df['value'] >= -0.33) & (tags_report_df['value'] <= 0.33)].shape[0]
positive = tags_report_df[tags_report_df['value'] > 0.33].shape[0]

print(f"  负面 (value < -0.33): {negative} ({negative/total*100:.1f}%)")
print(f"  中性 (-0.33 ≤ value ≤ 0.33): {neutral} ({neutral/total*100:.1f}%)")
print(f"  正面 (value > 0.33): {positive} ({positive/total*100:.1f}%)")

print(f"\n📄 TagsReport文件前5行:")
print(tags_report_df.head())

# 8. 保存原始预测结果到CSV（可选）
results_df = pd.DataFrame({
    'True_Label': true_labels,
    'Predicted_Label': [class_labels.get(c, f"未知({c})") for c in predicted_classes],
    'Predicted_Index': predicted_classes,
    'Confidence': [predicted_probas[i][predicted_classes[i]] for i in range(total)]
})

# 添加每个类别的概率
for i in range(len(class_labels)):
    results_df[f'Prob_{class_labels[i]}'] = predicted_probas[:, i]

results_df.to_csv('training_set_predictions.csv', index=False)
print("\n原始预测结果已保存到 'training_set_predictions.csv'")