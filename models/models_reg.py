from dataclasses import dataclass
from typing import Dict, Any

import torch

@dataclass
class RecommenderModel:
    name: str
    model_class: torch.nn.Module
    loss_function: torch.nn.Module
    params: Dict[str, Any]
    