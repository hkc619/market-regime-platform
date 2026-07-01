from dataclasses import dataclass
from typing import Any
import torch
from pathlib import Path
import numpy as np
import pandas as pd
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

## new inference dataclass for db 
@dataclass
class RawInferenceSeries:
    ticker: str

    ticker_close: pd.Series
    ticker_vol: pd.Series
    ticker_high: pd.Series
    ticker_low: pd.Series

    sup0_close: pd.Series
    sup1_close: pd.Series

    vix_s: pd.Series
    yr10_s: pd.Series
    yr2_s: pd.Series
    cpi_s: pd.Series

    idx: pd.Index

@dataclass
class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/models/"
    data_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/data/test_dataset.xlsx"


@dataclass
class ModelInputBundle:
    ticker: str

    latest_60_feat: np.ndarray
    latest_regime: int | None

    feature_rows: int
    # feature_dim: int

    start_date: pd.Timestamp
    end_date: pd.Timestamp


class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )


settings = Settings()