
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader

from helper_modules.metrics import NDCGMetric


class ModelTrainer:
    def __init__(self, model: torch.nn.Module, device: str, k: int = 5):
        self.model = model.to(device)
        self.device = device
        self._val_losses = []
        self.k = k
    
    def train_model(self, optimizer, loss_function, train_dataloader, valid_dataloader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for user, pos_item, neg_item in train_dataloader:
                user = user.to(self.device)
                pos_item = pos_item.to(self.device)
                neg_item = neg_item.to(self.device)

                output = self.model(user, pos_item, neg_item)
                loss = loss_function(*output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            ndcg_5, ndcg_10 = self.validate(valid_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}",
                  f"NDCG at 5 {ndcg_5:.4f}, NDCG at 10 {ndcg_10:.4f}")
            if self._is_early_stop(ndcg_5):
                print(f"Validation loss stopped improving. Aborting")
                break
        print(f"Number of parameters trained {self._get_n_params()}")
        return ndcg_5
    
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
    
    def validate(self, valid_dataloader: DataLoader) -> Tuple[float, float]:
        predictions, labels = self.predict(valid_dataloader)
        ndcg_at_5 = NDCGMetric().calculate_ndcg(predictions, labels, self.k)
        ndcg_at_10 = NDCGMetric().calculate_ndcg(predictions, labels, 10)
        return ndcg_at_5, ndcg_at_10

    def predict(self, valid_dataloader: DataLoader) -> Tuple[Dict[str, list[float]]]:
        """
        Predict the validation dataset. Returns:
        - dict with a list of predictions for each user
        - dict with a list of labels for each user
        """
        self.model.eval()
        predicted = {}
        true_relevance = {}

        with torch.no_grad():
            for user, pos_item, neg_item in valid_dataloader:
                user = user.to(self.device)
                pos_item = pos_item.to(self.device)
                neg_item = neg_item.to(self.device)

                user_emb, pos_item_emb, _, user_bias, pos_item_bias, _ = self.model(user, pos_item, neg_item)
                x = torch.sum(user_emb * pos_item_emb, dim=-1)
                x = x + pos_item_bias + user_bias
                preds = self.model.sigmoid(x)

                for uid, pred in zip(user, preds):
                    uid = uid.item()
                    if uid not in predicted:
                        predicted[uid] = []
                        true_relevance[uid] = []
                    predicted[uid].append(pred)
                    true_relevance[uid].append(1)

            for _ in range(20):
                for user, pos_item, neg_item in valid_dataloader:
                    user = user.to(self.device)
                    pos_item = pos_item.to(self.device)
                    neg_item = neg_item.to(self.device)
                    
                    user_emb, _, neg_item_emb, user_bias, _, neg_item_bias = self.model(user, pos_item, neg_item)
                    x = torch.sum(user_emb * neg_item_emb, dim=-1)
                    x = x + neg_item_bias + user_bias
                    preds = self.model.sigmoid(x)

                    for uid, pred in zip(user, preds):
                        uid = uid.item()
                        if uid not in predicted:
                            predicted[uid] = []
                            true_relevance[uid] = []
                        predicted[uid].append(pred)
                        true_relevance[uid].append(0)
        return predicted, true_relevance
