from dataclasses import dataclass, field
from torchmetrics.metric import Metric

@dataclass
class ClassificationMetric:
    name: str
    metric: Metric
    value: float = field(init=False)
    