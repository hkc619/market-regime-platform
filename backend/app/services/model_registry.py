import sys
import json
from pathlib import Path
from typing import Any

import torch
sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from ml.model import DualCNNGRUFusion

class ModelLoadError(Exception):
    pass

METADATA_PATH = Path("/Users/hkc619/Documents/PY/project/market-regime-platform/models/metadata.json")
## Using Path can check the file exist or not
def load_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    if not metadata_path.exists(): 
        raise ModelLoadError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def get_model_metadata(version: str) -> dict[str, Any]:
    metadata = load_metadata()

    for model_info in metadata.get("models", []):
        if model_info.get("version") == version:
            return model_info
    raise ModelLoadError(f"Model version not found: {version}")

def load_cnn_gru_model(version: str, device: str = "cpu"):
    model_info = get_model_metadata(version)

    checkpoint_path = Path(model_info["model_config"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise ModelLoadError(f"Checkpoint file not found: {checkpoint_path}")

    input_config = model_info["input_config"]
    num_features = input_config["num_total_features_per_timestep"]
    num_classes = model_info["num_classes"]

    model = DualCNNGRUFusion(
        in_features=num_features,
        n_classes=num_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return {
        "model": model,
        "metadata": model_info,
        "device": device,
        "version": version,
    }
