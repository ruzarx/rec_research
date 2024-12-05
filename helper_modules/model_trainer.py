
from typing import List
from collections import namedtuple

import torch
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, F1Score
from torchmetrics.metric import Metric

from metrics import ClassificationMetric

metrics_map = {'roc_auc': namedtuple(name="ROC AUC", metric=BinaryAUROC),
               'roc': namedtuple(name="ROC AUC", metric=BinaryAUROC),
               'precision': namedtuple(name="Precision", metric=BinaryPrecision),
               'recall': namedtuple(name="Recall", metric=BinaryRecall),
               'f1': namedtuple(name="F1 Score", metric=F1Score),
               'f1_score': namedtuple(name="F1 Score", metric=F1Score),}

class ModelTrainer:
    def __init__(self, model: torch.nn.Module, device: str, metrics: List[str]):
        self.model = model.to(device)
        self.device = device
        self.metrics = self._init_metrics(metrics)

    def _init_metrics(self, requested_metrics: List[str]) -> List[Metric]:
        metrics = []
        for metric in requested_metrics:
            if metric in metrics_map:
                current_metric = ClassificationMetric(name=metrics_map[metric].name)
                current_metric.metric = metrics_map[metric].metric()
                current_metric.metric = current_metric.metric.to(self.device)
                metrics.append(current_metric)
            else:
                print(f"{metric} is not available. You can have {metrics_map.keys()}")
        return metrics
    
    def train_model(self, optimizer, loss_function, train_dataloader, valid_dataloader, num_epochs):
        loss_function = loss_function.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch, labels in train_dataloader:
                labels = labels.to(self.device)
                out = self._predict(batch)
                loss = loss_function(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Train. Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_dataloader):.4f}")

            self.model.eval()
            val_loss = 0.0
            for batch, labels in valid_dataloader:
                out = self._predict(batch)
                labels = labels.to(self.device)
                loss = loss_function(out, labels)
                val_loss += loss.item()
                self._update_metrics(out, labels)
            print(f"Validation. Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss/len(valid_dataloader):.4f}")
            print(self._repr_metrics)
        print(f"Number of parameters trained {self._get_n_params()}")
        return

    def _reset_metrics(self):
        for metric in self.metrics:
            metric.metric.reset()
        return
    
    def _update_metrics(self, predictions: torch.tensor, labels: torch.tensor) -> None:
        for metric in self.metrics:
            metric.metric.update(predictions, labels)
        return
    
    def _compute_metrics(self) -> None:
        for metric in self.metrics:
            metric.value = metric.metric.compute()
        return
    
    def _repr_metrics(self) -> None:
        line = ''
        for metric in self.metrics:
            line += f"{metric.name}: {metric.value:.3f}, "
        return line[:-2]

    def _predict(self, batch) -> torch.tensor:
        (user_id, game_id, user_features, game_features) = batch
        user_id = user_id.to(self.device)
        game_id = game_id.to(self.device)
        user_features = user_features.to(self.device)
        game_features = game_features.to(self.device)
        predictions = self.model((user_id, game_id, user_features, game_features))
        return predictions
    
    def _get_n_params(self) -> int:
        n_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        return n_params
