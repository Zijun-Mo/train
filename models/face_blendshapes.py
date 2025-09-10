"""
TensorFlow Lite face_blendshapes模型包装器
"""
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import os


class FaceBlendshapesModel:
    """面部关键点特征提取模型"""
    
    def __init__(self, model_path: str):
        """
        初始化TFLite模型
        
        Args:
            model_path: TFLite模型文件路径
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _load_model(self):
        """加载TFLite模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建解释器
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出详情
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 验证模型输入输出形状
        expected_input_shape = [1, 146, 2]
        expected_output_shape = [52]
        
        actual_input_shape = self.input_details[0]['shape'].tolist()
        actual_output_shape = self.output_details[0]['shape'].tolist()
        
        if actual_input_shape != expected_input_shape:
            raise ValueError(f"输入形状不匹配，期望: {expected_input_shape}, 实际: {actual_input_shape}")
        
        if actual_output_shape != expected_output_shape:
            raise ValueError(f"输出形状不匹配，期望: {expected_output_shape}, 实际: {actual_output_shape}")
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        提取关键点特征
        
        Args:
            landmarks: 关键点坐标，形状为(146, 2)或(batch_size, 146, 2)
            
        Returns:
            特征向量，形状为(52,)或(batch_size, 52)
        """
        # 处理输入形状
        if landmarks.ndim == 2:
            # 单个样本，添加batch维度
            landmarks = landmarks[np.newaxis, ...]
            single_sample = True
        elif landmarks.ndim == 3:
            single_sample = False
        else:
            raise ValueError(f"输入形状不正确，期望(146, 2)或(batch_size, 146, 2)，实际: {landmarks.shape}")
        
        batch_size = landmarks.shape[0]
        features = []
        
        for i in range(batch_size):
            # 准备输入数据
            input_data = landmarks[i:i+1].astype(np.float32)
            
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 运行推理
            self.interpreter.invoke()
            
            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            features.append(output_data.flatten())
        
        features = np.array(features)
        
        if single_sample:
            return features[0]
        else:
            return features
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            'input_shape': self.input_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'output_dtype': str(self.output_details[0]['dtype']),
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }


class DualFaceBlendshapesExtractor:
    """双关键点特征提取器"""
    
    def __init__(self, model_path: str):
        """
        初始化双特征提取器
        
        Args:
            model_path: TFLite模型文件路径
        """
        self.model = FaceBlendshapesModel(model_path)
    
    def extract_features(self, expression_landmarks: np.ndarray, 
                        baseline_landmarks: np.ndarray) -> np.ndarray:
        """
        提取表情和基线关键点的特征并拼接
        
        Args:
            expression_landmarks: 表情关键点，形状为(146, 2)或(batch_size, 146, 2)
            baseline_landmarks: 基线关键点，形状为(146, 2)或(batch_size, 146, 2)
            
        Returns:
            拼接后的特征向量，形状为(104,)或(batch_size, 104)
        """
        # 提取表情特征
        expression_features = self.model.extract_features(expression_landmarks)
        
        # 提取基线特征
        baseline_features = self.model.extract_features(baseline_landmarks)
        
        # 拼接特征
        if expression_features.ndim == 1:
            # 单个样本
            combined_features = np.concatenate([expression_features, baseline_features])
        else:
            # 批量样本
            combined_features = np.concatenate([expression_features, baseline_features], axis=1)
        
        return combined_features


def test_face_blendshapes_model(model_path: str):
    """
    测试face_blendshapes模型
    
    Args:
        model_path: 模型文件路径
    """
    print(f"测试模型: {model_path}")
    
    try:
        # 创建模型实例
        model = FaceBlendshapesModel(model_path)
        
        # 打印模型信息
        info = model.get_model_info()
        print("模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试单个样本
        print("\n测试单个样本:")
        test_landmarks = np.random.randn(146, 2).astype(np.float32)
        features = model.extract_features(test_landmarks)
        print(f"输入形状: {test_landmarks.shape}")
        print(f"输出形状: {features.shape}")
        print(f"输出范围: [{features.min():.3f}, {features.max():.3f}]")
        
        # 测试批量样本
        print("\n测试批量样本:")
        batch_landmarks = np.random.randn(4, 146, 2).astype(np.float32)
        batch_features = model.extract_features(batch_landmarks)
        print(f"输入形状: {batch_landmarks.shape}")
        print(f"输出形状: {batch_features.shape}")
        
        # 测试双特征提取器
        print("\n测试双特征提取器:")
        dual_extractor = DualFaceBlendshapesExtractor(model_path)
        expression_landmarks = np.random.randn(146, 2).astype(np.float32)
        baseline_landmarks = np.random.randn(146, 2).astype(np.float32)
        combined_features = dual_extractor.extract_features(expression_landmarks, baseline_landmarks)
        print(f"表情关键点形状: {expression_landmarks.shape}")
        print(f"基线关键点形状: {baseline_landmarks.shape}")
        print(f"组合特征形状: {combined_features.shape}")
        
        print("\n模型测试完成！")
        
    except Exception as e:
        print(f"模型测试失败: {e}")


if __name__ == "__main__":
    # 测试模型
    model_path = "/home/jun/picture/extracted_models/face_blendshapes.tflite"
    test_face_blendshapes_model(model_path)
