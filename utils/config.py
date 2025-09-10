"""
配置文件解析工具
"""
import yaml
import os
from typing import Dict, Any
from datetime import datetime


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_experiment_dir()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_experiment_dir(self):
        """设置实验目录"""
        base_dir = self.config['logging']['save_dir']
        experiment_name = self.config['logging']['experiment_name']
        
        if experiment_name is None:
            # 自动生成实验名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 更新配置中的实验目录
        self.config['logging']['experiment_dir'] = experiment_dir
        
        # 保存配置文件到实验目录
        config_save_path = os.path.join(experiment_dir, 'config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get(self, key: str, default=None):
        """
        获取配置值，支持嵌套键
        
        Args:
            key: 配置键，支持'.'分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持'.'分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
    
    @property
    def device_config(self) -> Dict[str, Any]:
        """获取设备配置"""
        return self.config.get('device', {})
    
    def get_experiment_dir(self) -> str:
        """获取实验目录"""
        return self.config['logging']['experiment_dir']


def load_config(config_path: str = 'config.yaml') -> Config:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    return Config(config_path)
