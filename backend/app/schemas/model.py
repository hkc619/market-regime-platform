# backend/app/schemas/model.py

from pydantic import BaseModel
from typing import List, Dict, Any


class ModelInputConfig(BaseModel):
    window_size: int
    num_base_features: int
    num_delta_features: int
    num_total_features_per_timestep: int


class ModelConfig(BaseModel):
    architecture: str
    framework: str
    checkpoint_path: str


class TrainingScope(BaseModel):
    training_asset: str
    model_type: str
    pipeline_scope: str
    generalization_note: str | None = None


class MetricsSummary(BaseModel):
    accuracy: float
    macro_f1: float
    total_support: int


class ModelInfoResponse(BaseModel):
    version: str
    model_name: str
    task: str
    description: str | None = None
    num_classes: int
    classes: List[str]
    input_config: ModelInputConfig
    model_config: ModelConfig
    training_scope: TrainingScope
    metrics: Dict[str, Any]