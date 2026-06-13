import sys
import json
from pathlib import Path
from typing import Any
import torch

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from ml.model import DualCNNGRUFusion
from core.logging import get_logger

logger = get_logger("model_registry")

class ModelLoadError(Exception):
    pass

METADATA_PATH = Path("/Users/hkc619/Documents/PY/project/market-regime-platform/models/metadata.json")

## Using Path can check the file exist or not
def load_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    logger.info("Loading metadata | path=%s", metadata_path)

    if not metadata_path.exists(): 
        logger.error("Metadata file not found | path=%s", metadata_path)
        raise ModelLoadError(f"Metadata file not found: {metadata_path}")
    
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Metadata loaded successfully")
    return metadata
    
    
    
def get_model_metadata(version: str) -> dict[str, Any]:
    metadata = load_metadata()

    for model_info in metadata.get("models", []):
        if model_info.get("version") == version:
            logger.info("Model metadata found | version=%s", version)
            return model_info
        
    logger.error("Model version not found | version=%s", version)    
    raise ModelLoadError(f"Model version not found: {version}")

def load_cnn_gru_model(version: str, device: str = "cpu"):
    logger.info("Loading CNN-GRU model | version=%s | device=%s", version, device)
    
    model_info = get_model_metadata(version)

    checkpoint_path = Path(model_info["model_config"]["checkpoint_path"])

    if not checkpoint_path.exists():
        logger.error("Checkpoint file not found | path=%s", checkpoint_path)
        raise ModelLoadError(f"Checkpoint file not found: {checkpoint_path}")

    input_config = model_info["input_config"]
    num_features = input_config["num_total_features_per_timestep"]
    num_classes = model_info["num_classes"]


    logger.info(
        "Initializing model architecture | num_features=%s | num_classes=%s",
        num_features,
        num_classes,
    )

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
