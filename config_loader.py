from ruamel.yaml import YAML
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict

class DataConfig(BaseModel):
    image_size: int
    patch_size: int
    channels: int
    class_mapping: Dict[int, str] = Field(...)
    color_mapping: Dict[int, str] = Field(...)
    center_area_perc: float
    
class HyperparamsConfig(BaseModel):
    batch_size: int
    num_epochs: int
    learning_rate: float
    lr_patience: int
    lr_factor: float
    num_workers: int
    min_lr: float
    weight_decay: float
    smooth_factor: float
    diceloss_weight: float
    softcrossentropyloss_weight: float  
    early_stop_patience: int
    num_classes: int
    random_seed: int
    pin_memory: bool
    prefetch_factor: int

class PathsConfig(BaseModel):
    train_data_dir: str
    train_label_dir: str 
    val_data_dir: str
    val_label_dir: str
    test_data_dir: str
    test_label_dir: str
    evaluation_dir: str
    prediction_dir: str
    train_dir: str
    data_display_dir: str
    model_load_path: str

class Settings(BaseModel):
    data: DataConfig
    hyperparams: HyperparamsConfig
    paths: PathsConfig


def load_config():
    # 获取配置文件绝对路径
    config_path = Path(__file__).parent / "configs.yml"
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(config_path, 'r',encoding='utf-8') as f:
        config_data = yaml.load(f)
    
    return Settings(**config_data)

# 示例用法
if __name__ == '__main__':
    config = load_config()
    print(config.dict())