import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 你需要确保EEGModels.py在同一个目录，或者安装eegnet包
from EEGModels import EEGNet
import matplotlib.pyplot as plt

# 1. 加载CSV数据
df = pd.read_csv("emotions.csv")  # 修改为你的文件路径
print(f"数据集形状: {df.shape}")
print(f"前几列:\n{df.head()}")

# 2. 分离特征和标签
X = df.iloc[:, :-1].values  # 所有行，除最后一列的所有列（特征）
y = df.iloc[:, -1].values  # 所有行，最后一列（标签）

# 3. 编码标签：将 'NEGATIVE','NEUTRAL','POSITIVE' 转换为 0,1,2
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 现在y是数字
y_categorical = to_categorical(y_encoded)  # 转换为独热编码
num_classes = y_categorical.shape[1]
print(f"特征形状: {X.shape}, 标签形状: {y_categorical.shape}, 类别数: {num_classes}")

# 4. 标准化特征 (非常重要!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6. **关键修改：重塑数据以“模拟”EEGNet的输入形状**
# EEGNet期望输入形状为 (样本数, 通道数, 时间点, 1)
# 我们的CSV特征是 (样本数, 2549个特征)。我们需要将其重塑。
# 一个常见的做法是：将2549个特征“假装”成是 通道数 x 时间点 的序列。
# 例如，我们可以将特征重塑为 (样本数, 1, 2549, 1)，即1个通道，2549个时间点。
# 或者找一个能整除的数，比如 (样本数, 17, 150, 1) (因为17*150=2550，接近2549，需要先填充或截断)
# 这里为了简单，我们采用第一种方式：

X_train_reshaped = X_train[:, np.newaxis, :, np.newaxis]  # 形状: (样本数, 1, 2549, 1)
X_test_reshaped = X_test[:, np.newaxis, :, np.newaxis]  # 形状: (样本数, 1, 2549, 1)
print(f"重塑后的训练数据形状: {X_train_reshaped.shape}")

# 7. 修改并编译EEGNet模型
# 关键参数调整：Chans=1 (我们只有1个“通道”)， Samples=2549 (我们的特征长度)
model = EEGNet(
    nb_classes=num_classes,
    Chans=1,
    Samples=X_train_reshaped.shape[2],
    dropoutRate=0.5,
    kernLength=32,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()  # 打印模型结构，确认输入形状

# 8. 训练模型
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model_feature_csv.keras", save_best_only=True),
]

history = model.fit(
    X_train_reshaped,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,  # 从训练集中再分一部分作为验证集
    callbacks=callbacks,
    verbose=1,
)

# 9. 评估模型
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"\n测试集准确率: {accuracy:.4f}")

# 10. 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="训练准确率")
plt.plot(history.history["val_accuracy"], label="验证准确率")
plt.title("模型准确率")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="训练损失")
plt.plot(history.history["val_loss"], label="验证损失")
plt.title("模型损失")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_history_csv.png")
plt.show()

import joblib
from datetime import datetime

# 1. 保存模型 (你的代码里可能已有，但确保路径正确)
model.save("best_model_feature_csv.keras")
print("✅ 模型已保存.")

# 2. 【关键】保存标准化器 (scaler)
# 你需要有一个在训练时拟合好的 `scaler` 对象
# 它通常来自这两行代码：
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(); scaler.fit(X_train)
joblib.dump(scaler, "scaler.pkl")  # 保存scaler对象
print("✅ 标准化器 (scaler) 已保存为 'scaler.pkl'.")

# 3. (可选) 也保存标签编码器等
print("保存完成！现在可以运行 run_model.py 了。")
