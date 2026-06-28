from dataclasses import dataclass
from typing import Any
import torch
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

@dataclass
class DataConfig:
    data_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/data/dataset.xlsx"
    output_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/models/"

@dataclass
class TrainingConfig:
    DEVICE :Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_FRAC: float = 0.50 
    VAL_FRAC:   float = 0.25
    SEQ_LEN_M:  int = 60    # medium-term lookback (CNN-medium + GRU)
    SEQ_LEN_S:  int = 20 
    BATCH_SIZE: int = 64
    EPOCHS: int     = 60
    LR: float       = 1e-3
    PATIENCE: int   = 10
    SEED: int = 42

@dataclass
class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/models/"
    data_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/data/test_dataset.xlsx"

class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )


settings = Settings()