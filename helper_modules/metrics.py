from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from torchmetrics.metric import Metric

@dataclass
class ClassificationMetric:
    name: str
    metric: Metric = field(init=False)
    value: float = field(init=False)
    

class NDCGMetric:
    "Calculates NDCG at K for given predictions and true labels"

    def calculate_ndcg(self,
                       predictions: Dict[str, list[float]],
                       labels: Dict[str, list[float]],
                       k: int) -> float:
        ndcg_at_k = self._get_ndcg_at_k(predictions, labels, k)
        return ndcg_at_k

    def _get_dcg(self, relevance_scores: Dict[str, list[float]], k: int) -> float:
        "Calculate Discounted Cumulative Gain (DCG) at K"
        relevance_scores = relevance_scores[:k]
        gains = np.power(2, relevance_scores) - 1
        discounts = np.log2(np.arange(1, len(relevance_scores) + 1) + 1)
        return np.sum(gains / discounts)

    def _get_ndcg_at_k(self,
                       predictions: Dict[str, list[float]],
                       labels: Dict[str, list[float]],
                       k: int) -> float:
        "Calculate NDCG@K for each user and average across users"
        ndcg_scores = []
        for user in predictions.keys():
            # Sort true relevance based on predicted order
            user_predictions = np.array(predictions[user])
            user_labels = np.array(labels[user])

            sorted_indices = np.argsort(user_predictions)[::-1]
            sorted_true_relevance = user_labels[sorted_indices]

            dcg = self._get_dcg(sorted_true_relevance, k)

            ideal_relevance = np.sort(user_labels)[::-1]
            idcg = self._get_dcg(ideal_relevance, k)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        return np.mean(ndcg_scores)
