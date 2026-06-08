from dataclasses import dataclass
from typing import Any

@dataclass
class ModelState:
    model_loaded: bool = False
    model: Any = None
    metadata: dict | None = None
    version: str | None = None
    device: str = "cpu"
    error: str | None = None