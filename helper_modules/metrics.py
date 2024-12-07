from dataclasses import dataclass, field
from torchmetrics.metric import Metric

@dataclass
class ClassificationMetric:
    name: str
    metric: Metric = field(init=False)
    value: float = field(init=False)
    