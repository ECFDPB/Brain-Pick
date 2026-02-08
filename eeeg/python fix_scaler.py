# fix_scaler.py 的内容
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("正在修复...")
# 1. 加载数据
df = pd.read_csv('emotions.csv')

# 2. 【关键】只取前2548列作为特征，最后一列是标签，我们不要
X = df.iloc[:, :-1].values  # 这是正确的2548列特征
y = df.iloc[:, -1].values   # 这是标签列

print(f"✅ 正确分离：特征形状 {X.shape}， 标签形状 {y.shape}")

# 3. 用这2548列特征重新训练一个标准化器
scaler = StandardScaler()
scaler.fit(X)  # 这里只用了2548列！

# 4. 保存这个新的、正确的标准化器
joblib.dump(scaler, 'scaler_fixed.pkl')
print("✅ 新的、正确的标准化器已保存为 'scaler_fixed.pkl'")
print(f"   它现在期望的输入是 {scaler.n_features_in_} 列特征，不会再报错了。")