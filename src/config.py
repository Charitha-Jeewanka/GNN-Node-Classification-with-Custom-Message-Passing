import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class DatasetConfig:
    name: str
    root_dir: str

@dataclass
class ModelConfig:
    type: str
    hidden_dim: int
    num_layers: int
    dropout: float
    sage_aggregator: str

@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    device: str

@dataclass
class VisualizationConfig:
    subgraph_nodes: int
    seed_node: int
    output_dir: str

@dataclass
class AppConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from a YAML file and returns typed configuration objects."""
    with open(config_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    
    return AppConfig(
        dataset=DatasetConfig(**data["dataset"]),
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
        visualization=VisualizationConfig(**data["visualization"])
    )
