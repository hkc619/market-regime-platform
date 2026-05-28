from dataclasses import dataclass
import torch

@dataclass
class DataConfig:
    data_path: str = ""

@dataclass
class TrainingConfig:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")