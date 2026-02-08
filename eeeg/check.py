import pandas as pd
df = pd.read_csv('emotions.csv')
print(df.shape) # 应显示类似 (2132, 2549) 的形状
print(df.columns[-5:]) # 查看最后几列，确认标签列存在