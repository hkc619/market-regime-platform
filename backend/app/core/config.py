from dataclasses import dataclass
import torch

@dataclass
class DataConfig:
    data_path: str = "/Users/hkc619/Documents/PY/project/market-regime-platform/data/dataset.xlsx"

@dataclass
class TrainingConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_FRAC: float = 0.50 
    VAL_FRAC:   float = 0.25
    SEQ_LEN_M:  int = 60    # medium-term lookback (CNN-medium + GRU)
    SEQ_LEN_S:  int = 20 
    BATCH_SIZE: int = 64
    EPOCHS: int     = 60
    LR: float       = 1e-3
    PATIENCE: int   = 10