import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time
from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Union, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义 TagsReport 数据类
@dataclass
class TagsReport:
    username: str
    timestamp: int
    # Value will be a float number from -1.0 to 1.0, representing likeness.
    value: float

# 全局变量存储模型和标准化器
_model = None
_scaler = None

def load_models():
    """加载模型和标准化器（单例模式）"""
    global _model, _scaler
    
    if _model is None or _scaler is None:
        try:
            logger.info("正在加载模型和标准化器...")
            _model = load_model('best_model_feature_csv.keras')
            _scaler = joblib.load('scaler_fixed.pkl')
            logger.info("模型加载完成！")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    return _model, _scaler

def predict_from_csv(csv_path: str, has_label: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """从CSV文件读取数据进行预测"""
    try:
        df_new = pd.read_csv(csv_path)
        logger.info(f"从 {csv_path} 加载数据，形状: {df_new.shape}")
        
        if has_label:
            # 如果有标签列，分离特征和标签
            features = df_new.iloc[:, :-1].values  # 所有行，除了最后一列
            labels = df_new.iloc[:, -1].values  # 最后一列是标签
            return features, labels, df_new
        else:
            # 如果没有标签列，直接返回特征
            features = df_new.values
            return features, None, df_new
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise

def batch_predict(features_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对多行数据进行批量预测"""
    try:
        model, scaler = load_models()
        features_scaled = scaler.transform(features_array)
        features_reshaped = features_scaled[:, np.newaxis, :, np.newaxis]
        probas = model.predict(features_reshaped, verbose=0)
        classes = np.argmax(probas, axis=1)
        return classes, probas
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise

def map_prediction_to_value(pred_class: int, pred_proba: np.ndarray) -> float:
    """
    将模型预测映射到-1.0到1.0的范围
    使用概率加权：-1.0 * P(NEGATIVE) + 0.0 * P(NEUTRAL) + 1.0 * P(POSITIVE)
    """
    try:
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
    except Exception as e:
        logger.error(f"映射预测值时出错: {e}")
        return 0.0  # 返回中性值作为默认

def create_tags_report_from_predictions(
    predicted_classes: np.ndarray, 
    predicted_probas: np.ndarray,
    username_prefix: str = "user", 
    start_timestamp: Optional[int] = None
) -> List[TagsReport]:
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
    
    logger.info(f"创建了 {len(tags_reports)} 个TagsReport对象")
    return tags_reports

def export_tags_report_to_csv(tags_reports: List[TagsReport], output_file: str = 'tags_report.csv') -> pd.DataFrame:
    """
    将TagsReport列表导出为CSV文件
    """
    try:
        # 转换为字典列表
        data = [asdict(report) for report in tags_reports]
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 确保列顺序
        df = df[['username', 'timestamp', 'value']]
        
        # 保存为CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"已将TagsReport导出到 {output_file}")
        return df
    except Exception as e:
        logger.error(f"导出TagsReport到CSV失败: {e}")
        raise

def calculate_accuracy(predicted_classes: np.ndarray, true_labels: np.ndarray) -> Dict:
    """
    计算预测准确率和详细统计信息
    """
    class_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    label_to_idx = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    
    correct = 0
    total = len(true_labels)
    
    # 分类统计
    class_stats = {label: {'correct': 0, 'total': 0, 'accuracy': 0.0} for label in class_labels.values()}
    
    for i in range(total):
        # 如果标签是字符串，转换为数字
        if isinstance(true_labels[i], str):
            true_idx = label_to_idx.get(true_labels[i], -1)
        else:
            true_idx = int(true_labels[i])
        
        predicted_idx = predicted_classes[i]
        
        # 更新分类统计
        true_label_name = class_labels.get(true_idx, "UNKNOWN")
        if true_label_name in class_stats:
            class_stats[true_label_name]['total'] += 1
            if true_idx == predicted_idx:
                class_stats[true_label_name]['correct'] += 1
        
        # 更新总体统计
        if true_idx == predicted_idx:
            correct += 1
    
    # 计算准确率
    accuracy = correct / total * 100 if total > 0 else 0
    
    # 计算每个类别的准确率
    for label_name, stats in class_stats.items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total'] * 100
        else:
            stats['accuracy'] = 0.0
    
    return {
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'class_stats': class_stats
    }

def predict_emotions(
    data_source: Union[str, np.ndarray, pd.DataFrame],
    has_label: bool = True,
    username_prefix: str = "user",
    export_csv: bool = True,
    output_file: str = "tags_report.csv"
) -> Dict:
    """
    入口函数：对输入数据进行情感预测
    
    参数:
        data_source: 数据源，可以是以下之一：
            - 字符串: CSV文件路径
            - numpy数组: 特征数据
            - pandas DataFrame: 包含特征的数据框
        has_label: 数据是否包含标签列（仅当data_source是文件路径时有效）
        username_prefix: 用户名前缀
        export_csv: 是否导出CSV文件
        output_file: 输出CSV文件名
        
    返回:
        字典，包含:
            - success: 是否成功
            - tags_reports: TagsReport对象列表（转换为字典）
            - statistics: 统计信息
            - csv_path: 导出的CSV文件路径（如果export_csv=True）
            - message: 结果消息
    """
    try:
        logger.info(f"开始情感预测，数据源类型: {type(data_source)}")
        
        # 1. 准备数据
        true_labels = None
        if isinstance(data_source, str):
            # 如果是文件路径
            features, true_labels, _ = predict_from_csv(data_source, has_label)
        elif isinstance(data_source, pd.DataFrame):
            # 如果是DataFrame
            if has_label and 'label' in data_source.columns:
                features = data_source.drop('label', axis=1).values
                true_labels = data_source['label'].values
            else:
                features = data_source.values
        elif isinstance(data_source, np.ndarray):
            # 如果是numpy数组
            features = data_source
        else:
            raise ValueError(f"不支持的数据源类型: {type(data_source)}")
        
        logger.info(f"特征数据形状: {features.shape}")
        
        # 2. 批量预测
        predicted_classes, predicted_probas = batch_predict(features)
        logger.info(f"预测完成，共 {len(predicted_classes)} 个样本")
        
        # 3. 创建TagsReport
        tags_reports = create_tags_report_from_predictions(
            predicted_classes, predicted_probas, username_prefix
        )
        
        # 4. 计算统计信息
        stats = {
            'success': True,
            'total_samples': len(predicted_classes),
            'tags_reports': [asdict(report) for report in tags_reports],
            'message': f"成功预测 {len(predicted_classes)} 个样本"
        }
        
        # 如果有真实标签，计算准确率
        if true_labels is not None:
            accuracy_info = calculate_accuracy(predicted_classes, true_labels)
            stats['accuracy'] = accuracy_info['accuracy']
            stats['class_stats'] = accuracy_info['class_stats']
            stats['message'] += f"，准确率: {accuracy_info['accuracy']:.2f}%"
        
        # 5. 导出CSV
        if export_csv:
            df = export_tags_report_to_csv(tags_reports, output_file)
            
            # 添加value统计
            value_stats = {
                'min': float(df['value'].min()),
                'max': float(df['value'].max()),
                'mean': float(df['value'].mean()),
                'std': float(df['value'].std())
            }
            stats['value_stats'] = value_stats
            stats['csv_path'] = output_file
            
            # 情绪分布
            negative = df[df['value'] < -0.33].shape[0]
            neutral = df[(df['value'] >= -0.33) & (df['value'] <= 0.33)].shape[0]
            positive = df[df['value'] > 0.33].shape[0]
            total = len(df)
            
            stats['emotion_distribution'] = {
                'negative': {'count': negative, 'percentage': negative/total*100 if total > 0 else 0},
                'neutral': {'count': neutral, 'percentage': neutral/total*100 if total > 0 else 0},
                'positive': {'count': positive, 'percentage': positive/total*100 if total > 0 else 0}
            }
        
        logger.info(stats['message'])
        return stats
        
    except Exception as e:
        logger.error(f"情感预测失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"预测失败: {str(e)}"
        }

def predict_from_features(features_array: np.ndarray, **kwargs) -> Dict:
    """
    从特征数组进行预测（简化入口函数）
    """
    return predict_emotions(features_array, has_label=False, **kwargs)

def predict_from_dataframe(df: pd.DataFrame, has_label: bool = True, **kwargs) -> Dict:
    """
    从DataFrame进行预测（简化入口函数）
    """
    return predict_emotions(df, has_label=has_label, **kwargs)

# Flask API 示例
def create_flask_app():
    """
    创建一个Flask API服务，可供服务器端调用
    """
    try:
        from flask import Flask, request, jsonify
        app = Flask(__name__)
        
        # 预加载模型
        load_models()
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """健康检查端点"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': _model is not None,
                'scaler_loaded': _scaler is not None
            })
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """预测端点"""
            try:
                data = request.json
                
                if not data:
                    return jsonify({
                        'success': False,
                        'message': '请求数据为空'
                    }), 400
                
                # 检查数据格式
                if 'features' in data:
                    # 直接传递特征数组
                    features = np.array(data['features'])
                    result = predict_from_features(features)
                elif 'csv_path' in data:
                    # 传递CSV文件路径
                    csv_path = data['csv_path']
                    has_label = data.get('has_label', True)
                    username_prefix = data.get('username_prefix', 'user')
                    export_csv = data.get('export_csv', True)
                    output_file = data.get('output_file', 'tags_report.csv')
                    
                    result = predict_emotions(
                        csv_path,
                        has_label=has_label,
                        username_prefix=username_prefix,
                        export_csv=export_csv,
                        output_file=output_file
                    )
                else:
                    return jsonify({
                        'success': False,
                        'message': '请提供features或csv_path参数'
                    }), 400
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"API预测失败: {e}")
                return jsonify({
                    'success': False,
                    'message': f'预测失败: {str(e)}'
                }), 500
        
        @app.route('/batch_predict', methods=['POST'])
        def batch_predict_endpoint():
            """批量预测端点（上传CSV文件）"""
            try:
                if 'file' not in request.files:
                    return jsonify({
                        'success': False,
                        'message': '没有上传文件'
                    }), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({
                        'success': False,
                        'message': '文件名为空'
                    }), 400
                
                # 保存上传的文件
                upload_dir = 'uploads'
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file.filename)
                file.save(file_path)
                
                # 获取其他参数
                has_label = request.form.get('has_label', 'true').lower() == 'true'
                username_prefix = request.form.get('username_prefix', 'user')
                export_csv = request.form.get('export_csv', 'true').lower() == 'true'
                output_file = request.form.get('output_file', 'tags_report.csv')
                
                # 进行预测
                result = predict_emotions(
                    file_path,
                    has_label=has_label,
                    username_prefix=username_prefix,
                    export_csv=export_csv,
                    output_file=output_file
                )
                
                # 清理上传的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"批量预测失败: {e}")
                return jsonify({
                    'success': False,
                    'message': f'批量预测失败: {str(e)}'
                }), 500
        
        return app
        
    except ImportError:
        logger.warning("Flask未安装，无法创建API服务")
        return None

# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='情感分析模型预测')
    parser.add_argument('--csv', type=str, help='CSV文件路径')
    parser.add_argument('--no-label', action='store_true', help='数据不包含标签列')
    parser.add_argument('--output', type=str, default='tags_report.csv', help='输出CSV文件名')
    parser.add_argument('--prefix', type=str, default='user', help='用户名前缀')
    parser.add_argument('--api', action='store_true', help='启动API服务')
    parser.add_argument('--port', type=int, default=5000, help='API服务端口')
    
    args = parser.parse_args()
    
    if args.api:
        # 启动API服务
        app = create_flask_app()
        if app:
            logger.info(f"启动API服务，端口: {args.port}")
            app.run(host='0.0.0.0', port=args.port, debug=False)
        else:
            logger.error("无法启动API服务，请安装Flask: pip install flask")
    elif args.csv:
        # 命令行预测
        result = predict_emotions(
            args.csv,
            has_label=not args.no_label,
            username_prefix=args.prefix,
            export_csv=True,
            output_file=args.output
        )
        
        if result['success']:
            print(f"\n✅ 预测成功!")
            print(f"   样本数: {result['total_samples']}")
            if 'accuracy' in result:
                print(f"   准确率: {result['accuracy']:.2f}%")
            if 'csv_path' in result:
                print(f"   结果已保存: {result['csv_path']}")
            print(f"   消息: {result['message']}")
        else:
            print(f"\n❌ 预测失败: {result.get('message', '未知错误')}")
    else:
        print("请提供CSV文件路径或使用--api启动服务")
        parser.print_help()

if __name__ == "__main__":
    main()