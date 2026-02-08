import numpy as np
import pandas as pd

def create_test_data(num_samples=100, num_features=2548, csv_path="test_emotions.csv"):
    """
    创建与训练集维度一致的测试数据
    """
    # 生成随机特征数据（模拟真实数据分布）
    # 假设特征值在 [-1000, 1000] 范围内，类似EEG数据
    features = np.random.uniform(-1000, 1000, size=(num_samples, num_features))
    
    # 生成随机标签（三个类别）
    labels = np.random.choice(['NEGATIVE', 'NEUTRAL', 'POSITIVE'], size=num_samples)
    
    # 创建列名（与训练集保持一致很重要！）
    # 如果知道训练集列名，最好使用相同的列名
    feature_columns = [f'feature_{i}' for i in range(num_features)]
    
    # 创建DataFrame
    df = pd.DataFrame(features, columns=feature_columns)
    df['label'] = labels
    
    # 保存到CSV
    df.to_csv(csv_path, index=False)
    
    print(f"测试数据已创建并保存到: {csv_path}")
    print(f"测试数据形状: {df.shape}")
    print(f"标签分布: {df['label'].value_counts().to_dict()}")
    
    return df

# 使用示例
test_df = create_test_data(num_samples=99, num_features=2548, csv_path="test_emotions_correct.csv")