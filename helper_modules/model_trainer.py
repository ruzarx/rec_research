
from typing import List
import numpy as np
from collections import namedtuple

import torch
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score

from helper_modules.metrics import ClassificationMetric

metric_descriptor = namedtuple("metric_descriptor", ["name", "metric"])
metrics_map = {'roc_auc': metric_descriptor(name="ROC AUC", metric=BinaryAUROC),
               'roc': metric_descriptor(name="ROC AUC", metric=BinaryAUROC),
               'precision': metric_descriptor(name="Precision", metric=BinaryPrecision),
            #    'precision_at_k': metric_descriptor(name="Precision", metric=BinaryPrecision),
               'recall': metric_descriptor(name="Recall", metric=BinaryRecall),
               'f1': metric_descriptor(name="F1 Score", metric=BinaryF1Score),
               'f1_score': metric_descriptor(name="F1 Score", metric=BinaryF1Score),}

class ModelTrainer:
    def __init__(self, model: torch.nn.Module, device: str, metrics: List[str], k: int = 5):
        self.model = model.to(device)
        self.device = device
        self.metrics = self._init_metrics(metrics)
        self._val_losses = []
        self.k = k

    def _init_metrics(self, requested_metrics: List[str]) -> List[ClassificationMetric]:
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}", end=' ')

            # self.model.eval()
            # val_loss = 0.0
            # self._reset_metrics()
            # for batch, labels in valid_dataloader:
            #     out = self._predict(batch)
            #     labels = labels.to(self.device)
            #     loss = loss_function(out, labels)
            #     val_loss += loss.item()
            #     self._update_metrics(out, labels)
            # mean_val_loss = val_loss / len(valid_dataloader)
            ndcg_5, ndcg_10 = self.validate(valid_dataloader, k=5)
            # ndcg_10 = self.validate(valid_dataloader, k=10)
            # precision_at_5 = self._validate(valid_dataloader, k=5)
            # precision_at_10 = self._validate(valid_dataloader, k=10)
            print(f"NDCG at 5 {ndcg_5:.4f}, NDCG at 10 {ndcg_10:.4f}")
            # print(self._repr_metrics())
            if self._is_early_stop(ndcg_5):
                print(f"Validation loss stopped improving. Aborting")
                break
        print(f"Number of parameters trained {self._get_n_params()}")
        return ndcg_5

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
        self._compute_metrics()
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
    
    def _is_early_stop(self, current_loss: float) -> bool:
        if len(self._val_losses) <= 2:
            self._val_losses.append(current_loss)
        else:
            if current_loss < self._val_losses[-1] < self._val_losses[-2]:
                return True
            self._val_losses.append(current_loss)
        return False
        
    def dcg_torch(self, relevance_scores, k):
        """
        Calculate Discounted Cumulative Gain (DCG) at K.
        """
        relevance_scores = relevance_scores[:k]
        gains = torch.pow(2, relevance_scores) - 1
        discounts = torch.log2(torch.arange(1, len(relevance_scores) + 1, dtype=torch.float32) + 1)
        return torch.sum(gains / discounts)

    def ndcg_user_wise_torch(self, predicted, true_relevance, k):
        """
        Calculate NDCG@K for each user and average across users.
        """
        ndcg_scores = []

        for user in predicted.keys():
            # Sort true relevance based on predicted order
            user_predicted_scores = predicted[user]
            user_true_relevance = true_relevance[user]

            sorted_indices = torch.argsort(user_predicted_scores, descending=True)
            sorted_true_relevance = user_true_relevance[sorted_indices]

            # Calculate DCG@K
            dcg = self.dcg_torch(sorted_true_relevance, k)

            # Calculate IDCG@K (ideal ranking)
            ideal_relevance = torch.sort(user_true_relevance, descending=True).values
            idcg = self.dcg_torch(ideal_relevance, k)

            # Normalize
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        # Return mean NDCG across all users
        return torch.mean(torch.tensor(ndcg_scores))

    def validate(self, valid_dataloader, k):
        """
        Validate the model and calculate NDCG@K using the validation dataloader.
        """
        self.model.eval()  # Set model to evaluation mode
        predicted = {}
        true_relevance = {}

        with torch.no_grad():  # No gradients required for validation
            for batch, labels in valid_dataloader:
                # Unpack the batch
                user_id, game_id, user_features, game_features = batch

                # Move data to device
                user_id, game_id = user_id.to(self.device), game_id.to(self.device)
                user_features, game_features = user_features.to(self.device), game_features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass to get predictions
                outputs = self.model(batch)

                # Store predictions and true relevance by user
                for uid, gid, pred, label in zip(user_id, game_id, outputs, labels):
                    if uid.item() not in predicted:
                        predicted[uid.item()] = []
                        true_relevance[uid.item()] = []
                    predicted[uid.item()].append(pred.item())
                    true_relevance[uid.item()].append(label.item())

        # Convert lists to tensors for each user
        for user in predicted.keys():
            predicted[user] = torch.tensor(predicted[user])
            true_relevance[user] = torch.tensor(true_relevance[user])

        # Calculate NDCG@K
        mean_ndcg_5 = self.ndcg_user_wise_torch(predicted, true_relevance, 5)
        mean_ndcg_10 = self.ndcg_user_wise_torch(predicted, true_relevance, 10)
        return mean_ndcg_5, mean_ndcg_10

    
    # def precision_at_k_pytorch(self, user_ids, preds, targets, k):
    #     """
    #     Calculate Precision@K in PyTorch.

    #     Args:
    #         user_ids (torch.Tensor): Tensor of user IDs (shape: [num_samples]).
    #         game_ids (torch.Tensor): Tensor of game IDs (not used for Precision@K but useful for other metrics).
    #         preds (torch.Tensor): Tensor of predicted scores (shape: [num_samples]).
    #         targets (torch.Tensor): Tensor of binary relevance labels (1 if relevant, 0 otherwise) (shape: [num_samples]).
    #         k (int): Number of top predictions to consider for Precision@K.
            
    #     Returns:
    #         float: Mean Precision@K across all users.
    #     """
    #     # Combine user IDs, predictions, and relevance into a single tensor
    #     data = torch.stack((user_ids, preds, targets), dim=1)
        
    #     # Sort by user ID and then by predicted scores in descending order
    #     sorted_data = data[data[:, 1].argsort(descending=True)]
    #     sorted_data = sorted_data[sorted_data[:, 0].argsort()]
        
    #     # Group data by user
    #     unique_users = torch.unique(user_ids)
    #     precisions = []

    #     for user in unique_users:
    #         # Extract predictions for the current user
    #         user_data = sorted_data[sorted_data[:, 0] == user]
            
    #         # Select the top-K predictions
    #         top_k_data = user_data[:k]
            
    #         # Calculate Precision@K
    #         relevant_items = top_k_data[:, 2].sum().item()  # Sum of relevant items in top-K
    #         precisions.append(relevant_items / k)
        
    #     # Return mean Precision@K across all users
    #     return sum(precisions) / len(precisions)
    
