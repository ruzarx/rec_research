
from typing import Tuple, List, Optional
import json
import logging
from torch.utils.data import DataLoader
import pandas as pd

from helper_modules.dataset_class import BPRDataset, FeaturesCounts


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    

class SteamDataProcessorClean:
    def __init__(self,
                 batch_size: int,
                 min_user_interactions: int,
                 min_game_interactions: int,
                 n_valid_sample: int,
                 ):
        self.data_path = f"data/steam_games/"
        self.data_file_name = "steam-200k.csv"
        self.batch_size = batch_size
        self.user_column_name = 'user'
        self.item_column_name = 'game'
        self.label_column_name = 'hours'
        self.n_valid_sample = n_valid_sample
        self.min_user_interactions = min_user_interactions
        self.min_game_interactions = min_game_interactions
        self.features_stats = FeaturesCounts()
        return
    
    def get_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader, FeaturesCounts]:
        data, features_stats = self._prepare_raw_data()
        train_df, valid_df, test_df = self._split_train_valid(data)
        num_items = data['game'].nunique()
        train_user_item_pairs = []
        for _, user, item in train_df[train_df['hours'] == 1][['user', 'game']].itertuples():
            train_user_item_pairs.append((user, item))
        print(f"N train interactions {len(train_user_item_pairs)}")
        valid_user_item_pairs = []
        for _, user, item in valid_df[valid_df['hours'] == 1][['user', 'game']].itertuples():
            valid_user_item_pairs.append((user, item))
        print(f"N valid interactions {len(valid_user_item_pairs)}")
        test_user_item_pairs = []
        for _, user, item in test_df[test_df['hours'] == 1][['user', 'game']].itertuples():
            test_user_item_pairs.append((user, item))
        print(f"N test interactions {len(test_user_item_pairs)}")
        train_dataset = BPRDataset(train_user_item_pairs, num_items)
        valid_dataset = BPRDataset(valid_user_item_pairs, num_items)
        test_dataset = BPRDataset(test_user_item_pairs, num_items)
        train_dataloader = BPRDataset.get_dataloader(train_dataset, self.batch_size)
        valid_dataloader = BPRDataset.get_dataloader(valid_dataset, self.batch_size)
        test_dataloader = BPRDataset.get_dataloader(test_dataset, self.batch_size)
        logging.info(f"Created dataloaders for train {len(train_dataloader)}, validation {len(valid_dataloader)}, test {len(test_dataloader)}")
        return train_dataloader, valid_dataloader, test_dataloader, features_stats
    
    def _prepare_raw_data(self) -> Tuple[pd.DataFrame, FeaturesCounts]:
        df = self._load_raw_data()
        df = self._filter_duplicates(df)
        df = self._filter_min_interactions(df)
        df = self._define_validation(df)
        df = self._keep_purchases_only(df)
        df = self._encode_entity(df, 'user')
        df = self._encode_entity(df, 'game')
        features_stats = self._collect_feature_stats(df, None, None)
        return df, features_stats

    def _load_raw_data(self) -> pd.DataFrame:
        file_path = self.data_path + self.data_file_name
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(
                    file_path,
                    names=['user', 'game', 'action', 'hours', 'sic'],
                    header=None
                    ).drop(
                        columns='sic'
                        ).drop_duplicates()
            logger.info("Data loaded successfully with shape: %s", df.shape)
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
    
    def _filter_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        original_shape = data.shape[0]
        data = data.groupby(['user', 'game', 'action'], as_index=False)['hours'].sum()
        logging.info(f"Combined duplicates from {original_shape} to {data.shape[0]}")
        return data
    
    def _encode_entity(self,
                       data: pd.DataFrame,
                       entity_column: str
                       ) -> Tuple[pd.DataFrame]:
        encoding = dict()
        rev_encoding = dict()
        counter = 0
        data[entity_column] = data[entity_column].astype(str)
        for entity in data[entity_column].unique():
            encoding[entity] = counter
            rev_encoding[counter] = entity
            counter += 1
        data[entity_column] = data[entity_column].map(encoding).fillna(-1)
        logging.info(f"Encoded {entity_column}, {len(encoding)} unique values")
        self._store_encoding_data(rev_encoding, f"{entity_column}_encoding")
        return data
    
    def _filter_min_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        original_shape = data.shape[0]
        games_count = data['game'].value_counts()
        good_games = games_count[games_count >= self.min_game_interactions].index
        data = data[data['game'].isin(good_games)]

        users_count = data['user'].value_counts()
        good_users = users_count[users_count >= self.min_user_interactions].index
        data = data[data['user'].isin(good_users)]
        logging.info(f"Dropped rare games from {games_count.shape[0]} to {len(good_games)}")
        logging.info(f"Dropped inactive users from {users_count.shape[0]} to {len(good_users)}")
        logging.info(f"Overall dropped dataset from {original_shape} to {data.shape[0]}")
        return data
    
    def _define_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        data['n_event'] = data.groupby('user', as_index=False).cumcount() + 1
        data['is_valid'] = False
        data['is_test'] = False
        data.loc[
            data[data['action'] == 'purchase'].sort_values(
                ['user', 'n_event']
                ).groupby('user').tail(self.n_valid_sample).index,
            'is_valid'] = True
        data.loc[
            data[(data['action'] == 'purchase') & (data['is_valid'] == False)].sort_values(
                ['user', 'n_event']
                ).groupby('user').tail(self.n_valid_sample).index,
            'is_test'] = True
        data = data.drop(columns='n_event')
        logging.info(f"Added validation colums. "
                     f" Train samples {data[(data['is_valid'] == False) & (data['is_test'] == False)].shape[0]}"
                     f" Validation samples {data[data['is_valid'] == True].shape[0]}"
                     f" Test samples {data[data['is_test'] == True].shape[0]}"
                     )
        return data
    
    def _get_entity_columns_names(self, data: pd.DataFrame) -> List[str]:
        feature_cols = [col for col in data.columns if col not in ['user', 'game', 'hours']]
        logging.info(f"Feature columns: {feature_cols}")
        return feature_cols
    
    def _keep_purchases_only(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data['action'] == 'purchase'].reset_index(drop=True)
        logging.info(f"Dropped play data: {data.shape}")
        return data
    
    def _split_train_valid(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = data[(data['is_valid'] == False) & (data['is_test'] == False)].reset_index(drop=True).drop(columns=['is_valid', 'is_test'])
        valid_df = data[data['is_valid'] == True].reset_index(drop=True).drop(columns=['is_valid', 'is_test'])
        test_df = data[data['is_test'] == True].reset_index(drop=True).drop(columns=['is_valid', 'is_test'])
        return train_df, valid_df, test_df
    
    def _store_encoding_data(self, values: dict, name: str) -> None:
        with open(f"{self.data_path}{name}.json", 'w') as file:
            json.dump(values, file)
            logging.info(f"Saved {name} to {self.data_path}{name}.json")
        return
    
    def _collect_feature_stats(self,
                               data: pd.DataFrame,
                               user_features: Optional[List[str]],
                               item_features: Optional[List[str]]
                               ) -> FeaturesCounts:
        feature_stats = FeaturesCounts()
        if user_features and item_features:
            feature_stats.user_features = user_features
            feature_stats.item_features = item_features
            feature_stats.n_user_features = len(user_features)
            feature_stats.n_item_features = len(item_features)
        feature_stats.n_users = data['user'].nunique()
        feature_stats.n_items = data['game'].nunique()
        return feature_stats
    