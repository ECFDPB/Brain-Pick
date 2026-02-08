# emotion_predictor/__init__.py
"""
情感分析模型包
提供从EEG/生理信号预测用户情感和喜好的功能

主要功能：
1. 加载预训练模型进行情感预测
2. 将预测结果转换为标准格式
3. 提供API接口供服务器调用
"""

__version__ = "1.0.0"
__author__ = "Your Team"
__email__ = "your.email@example.com"

import logging

# 配置包级别的日志
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 从各个模块导入主要功能
try:
    from .predictor import (
        TagsReport,
        predict_emotions,
        predict_from_features,
        predict_from_dataframe,
        batch_predict,
        load_models
    )
    
    from .api import create_flask_app, create_fastapi_app
    
    # 定义 __all__ 控制 from emotion_predictor import * 的行为
    __all__ = [
        'TagsReport',
        'predict_emotions',
        'predict_from_features', 
        'predict_from_dataframe',
        'batch_predict',
        'load_models',
        'create_flask_app',
        'create_fastapi_app'
    ]
    
    print(f"✅ 情感分析包 v{__version__} 已加载")
    
except ImportError as e:
    print(f"⚠️  导入包时出错: {e}")
    # 可以设置占位符或提供降级方案