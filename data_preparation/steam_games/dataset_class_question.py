'Dataclass and dataloader for Steam Games dataset'

from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from helper_modules.data_types import dataset_item_type


class SteamDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 user_column_name: str,
                 item_column_name: str,
                 label_column_name: str,
                 user_feature_names: List[str] = None,
                 item_feature_names: List[str] = None,
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
        if self.user_feature_names and self.item_feature_names:
            user_features = torch.tensor(row[self.user_feature_names].tolist(), dtype=torch.float32)
            item_features = torch.tensor(row[self.item_feature_names].tolist(), dtype=torch.float32)
        else:
            user_features, item_features = None, None
        labels = torch.tensor(row[self.label_column_name], dtype=torch.float32)
        return (user_id, item_id, user_features, item_features), labels
    
    def __len__(self):
        return self.data.shape[0]
    

def get_data_loader(dataset: Dataset, batch_size: int = 64) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader
