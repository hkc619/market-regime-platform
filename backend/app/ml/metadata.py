# app/ml/metadata.py

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


CLASS_NAMES = [
    "Trending-Up",
    "Transition-Up",
    "Transition-Down",
    "Trending-Down",
]


def build_model_metadata_entry(
    *,
    version: str,
    model_name: str,
    checkpoint_path: str,
    scaler_path: str | None,
    target_asset: str,
    assets: list[str],
    raw_data_start_date: date | str,
    raw_data_end_date: date | str,
    training_rows: int,
    feature_rows: int,
    test_rows: int | None,
    metrics: dict[str, Any],
    window_size: int = 60,
    num_base_features: int = 56,
    num_delta_features: int = 17,
    num_total_features_per_timestep: int = 73,
    required_raw_days: int = 312,
) -> dict[str, Any]:
    model_config = {
        "architecture": "CNN-GRU",
        "framework": "PyTorch",
        "checkpoint_path": checkpoint_path,
    }

    if scaler_path:
        model_config["scaler_path"] = scaler_path

    return {
        "version": version,
        "model_name": model_name,
        "task": "Market Regime Classification",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "class_index_mapping": {
            str(i): class_name for i, class_name in enumerate(CLASS_NAMES)
        },
        "assets": assets,
        "data_sources": {
            "type": "postgresql",
            "price_table": "market_prices",
            "macro_daily_table": "macro_daily",
            "macro_monthly_table": "macro_monthly",
            "target_asset": target_asset,
            "context_assets": [asset for asset in assets if asset != target_asset],
        },
        "input_config": {
            "window_size": window_size,
            "target_asset": target_asset,
            "num_base_features": num_base_features,
            "num_delta_features": num_delta_features,
            "num_total_features_per_timestep": num_total_features_per_timestep,
            "required_raw_days": required_raw_days,
            "notes": (
                "Base features are generated before sequence construction. "
                "Delta features are one-step changes of selected informative "
                "features and are stacked onto the input sequence at sequence-build time."
            ),
        },
        "model_config": model_config,
        "training_config": {
            "train_split": "chronological",
            "shuffle": False,
            "lookahead_bias_control": True,
            "data_source": "postgresql",
        },
        "training_data": {
            "raw_data_start_date": str(raw_data_start_date),
            "raw_data_end_date": str(raw_data_end_date),
            "training_rows": training_rows,
            "feature_rows": feature_rows,
            "test_rows": test_rows,
        },
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).date().isoformat(),
        "notes": "DB-backed CNN-GRU model export for market regime classification.",
    }


def append_model_metadata(
    metadata_path: str | Path,
    model_entry: dict[str, Any],
) -> None:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {"models": []}

    metadata.setdefault("models", [])
    metadata["models"].append(model_entry)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)