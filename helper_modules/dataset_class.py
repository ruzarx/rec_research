'A versatile class for the research recsystem dataclass'

from dataclasses import dataclass, field
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from helper_modules.data_types import dataset_item_type


@dataclass
class FeaturesCounts:
    n_users: int = field(init=False)
    n_items: int = field(init=False)
    user_features: list = field(init=False)
    item_features: list = field(init=False)
    n_user_features: int = field(init=False)
    n_item_features: int = field(init=False)
    

class RecDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 user_column_name: str,
                 item_column_name: str,
                 label_column_name: str,
                 user_feature_names: List[str],
                 item_feature_names: List[str],
                 ) -> None:
        self.data = data
        self.user_column_name = user_column_name
        self.item_column_name = item_column_name
        self.label_column_name = label_column_name
        self.user_feature_names = user_feature_names
        self.item_feature_names = item_feature_names
        return

    def __getitem__(self, index: int) -> dataset_item_type:
        row = self.data.iloc[index]
        user_id = row[self.user_column_name]
        item_id = row[self.item_column_name]
        user_features = torch.tensor(row[self.user_feature_names].tolist(), dtype=torch.float32)
        item_features = torch.tensor(row[self.item_feature_names].tolist(), dtype=torch.float32)
        labels = torch.tensor(row[self.label_column_name], dtype=torch.float32)
        return (user_id, item_id, user_features, item_features), labels
    
    def __len__(self):
        return self.data.shape[0]
    
    @staticmethod
    def get_data_loader(dataset: Dataset, batch_size: int) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return dataloader
