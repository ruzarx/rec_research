
import json
import logging
from typing import Tuple, List, Optional, Dict
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from helper_modules.dataset_class import BPRDataset, FeaturesCounts


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    

class NetflixDataProcessorClean:
    def __init__(self,
                 batch_size: int,
                 min_user_interactions: int,
                 min_item_interactions: int,
                 n_valid_sample: int):
        self.data_path = f"data/netflix_movies/"
        self.data_folder_name = "training_set"
        self.batch_size = batch_size
        self.user_column_name = 'user'
        self.item_column_name = 'movie'
        self.label_column_name = 'rating'
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.n_valid_sample = n_valid_sample
        self.features_stats = FeaturesCounts()
        return
    
    def get_dataset(self, n_unique_users: int) -> Tuple[DataLoader, DataLoader, DataLoader, FeaturesCounts]:
        data, features_stats = self._prepare_raw_data(n_unique_users)
        train_df, valid_df, test_df = self._split_train_valid(data)
        num_items = data[data[self.label_column_name] >= 4][self.item_column_name].nunique()
        train_pos_user_item_pairs = self._get_pos_user_item_pairs(train_df)
        train_neg_user_item_pairs = self._get_neg_user_item_pairs(train_df)
        print(f"N train positive interactions {len(train_pos_user_item_pairs)}")
        print(f"Negative users: {len(train_neg_user_item_pairs)}, negative interactions: {sum([len(x) for x in train_neg_user_item_pairs.values()])}")
        valid_pos_user_item_pairs = self._get_pos_user_item_pairs(valid_df)
        valid_neg_user_item_pairs = self._get_neg_user_item_pairs(valid_df)
        print(f"N valid interactions: positive {len(valid_pos_user_item_pairs)}")
        print(f"Negative users: {len(valid_neg_user_item_pairs)}, negative interactions: {sum([len(x) for x in valid_neg_user_item_pairs.values()])}")
        test_pos_user_item_pairs = self._get_pos_user_item_pairs(test_df)
        test_neg_user_item_pairs = self._get_neg_user_item_pairs(test_df)
        print(f"N test interactions: positive {len(test_pos_user_item_pairs)}")
        print(f"Negative users: {len(test_neg_user_item_pairs)}, negative interactions: {sum([len(x) for x in test_neg_user_item_pairs.values()])}")
        train_dataset = BPRDataset(train_pos_user_item_pairs, train_neg_user_item_pairs, num_items)
        valid_dataset = BPRDataset(valid_pos_user_item_pairs, valid_neg_user_item_pairs, num_items)
        test_dataset = BPRDataset(test_pos_user_item_pairs, test_neg_user_item_pairs, num_items)
        train_dataloader = BPRDataset.get_dataloader(train_dataset, self.batch_size)
        valid_dataloader = BPRDataset.get_dataloader(valid_dataset, self.batch_size)
        test_dataloader = BPRDataset.get_dataloader(test_dataset, self.batch_size)
        logging.info(f"Created dataloaders for train {len(train_dataloader)}, validation {len(valid_dataloader)}, test {len(test_dataloader)}")
        return train_dataloader, valid_dataloader, test_dataloader, features_stats
    
    def _get_pos_user_item_pairs(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        condition = data[self.label_column_name] >= 4
        current_data = data[condition][[self.user_column_name, self.item_column_name]]
        users = current_data[self.user_column_name].values
        items = current_data[self.item_column_name].values
        return [(user, item) for user, item in zip(users, items)]
    
    def _get_neg_user_item_pairs(self, data: pd.DataFrame) -> Dict[int, List[int]]:
        condition = data[self.label_column_name] <= 2
        current_data = data[condition][[self.user_column_name, self.item_column_name]]
        users = current_data[self.user_column_name].values
        items = current_data[self.item_column_name].values
        user_item_neg_interations = dict()
        for user, item in zip(users, items):
            if user in user_item_neg_interations:
                user_item_neg_interations[user].append(item)
            else:
                user_item_neg_interations[user] = [item]
        return user_item_neg_interations
    
    def _prepare_raw_data(self, n_unique_users: int) -> Tuple[pd.DataFrame, FeaturesCounts]:
        df = self._load_raw_data()
        if n_unique_users is not None:
            df = self._choose_random_users(df, n_unique_users)
        df = self._filter_duplicates(df)
        df = self._filter_min_interactions(df)
        df = self._define_validation(df)
        df = self._encode_entity(df, self.user_column_name)
        df = self._encode_entity(df, self.item_column_name)
        features_stats = self._collect_feature_stats(df, None, None)
        return df, features_stats

    def _load_raw_data(self) -> pd.DataFrame:
        file_path = self.data_path + self.data_folder_name
        logger.info(f"Loading data from {file_path}")
        try:
            all_movies, all_users, all_ratings, all_dates = [], [], [], []
            for file_name in Path(file_path).iterdir():
                if file_name.suffix == '.txt':
                    movie_data = self._read_single_file(file_name)
                    movie_id = list(movie_data.keys())[0]
                    users, ratings, dates = movie_data[movie_id]
                    all_movies.extend([movie_id] * len(users))
                    all_users.extend(users)
                    all_ratings.extend(ratings)
                    all_dates.extend(dates)
            df = pd.DataFrame({self.item_column_name: list(map(int, all_movies)),
                                self.user_column_name: list(map(int, all_users)),
                                self.label_column_name: list(map(int, all_ratings)),
                                'date': pd.to_datetime(all_dates)})
            logger.info("Data loaded successfully with shape: %s", df.shape)
            return df
        except FileNotFoundError as e:
            logger.error(f"Path not found: {file_path}")
            raise e
        
    def _choose_random_users(self, df: pd.DataFrame, n_unique_users: int) -> pd.DataFrame:
        all_users = df[self.user_column_name].unique()
        random_users = np.random.choice(all_users, n_unique_users)
        df = df[df[self.user_column_name].isin(random_users)].reset_index(drop=True)
        return df
        
    def _read_single_file(self, file_name: Path) -> Dict[str, Tuple[List[str]]]:
        users = []
        ratings = []
        dates = []
        with open(file_name, 'r') as f:
            movie_id = f.readline().split(':')[0]
            for line in f:
                user_id, rating, date = line.split(',')
                users.append(user_id)
                ratings.append(rating)
                dates.append(date)
            movie_data = {movie_id: (users, ratings, dates)}
        return movie_data

    
    def _filter_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        original_shape = data.shape[0]
        data = data.drop_duplicates([self.user_column_name, self.item_column_name]).reset_index(drop=True)
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
        items_count = data[self.item_column_name].value_counts()
        good_items= items_count[items_count >= self.min_item_interactions].index
        data = data[data[self.item_column_name].isin(good_items)]

        users_count = data[self.user_column_name].value_counts()
        good_users = users_count[users_count >= self.min_user_interactions].index
        data = data[data[self.user_column_name].isin(good_users)]
        logging.info(f"Dropped rare items from {items_count.shape[0]} to {len(good_items)}")
        logging.info(f"Dropped inactive users from {users_count.shape[0]} to {len(good_users)}")
        logging.info(f"Overall dropped dataset from {original_shape} to {data.shape[0]}")
        return data
    
    def _define_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.label_column_name] = data[self.label_column_name].astype(int)
        condition = data[self.label_column_name] >= 4
        data['is_valid'] = False
        data['is_test'] = False
        data.loc[
            data[condition].sort_values([self.user_column_name, 'date']).groupby(
                self.user_column_name
                ).tail(self.n_valid_sample).index,
            'is_valid'] = True
        data.loc[
            data[condition & (data['is_valid'] == False)].sort_values(
                [self.user_column_name, 'date']
                ).groupby(
                    self.user_column_name
                    ).tail(self.n_valid_sample).index,
            'is_test'] = True
        data = data.drop(columns='date')
        logging.info(f"Added validation colums. "
                     f" Train samples {data[(data['is_valid'] == False) & (data['is_test'] == False)].shape[0]}"
                     f" Validation samples {data[data['is_valid'] == True].shape[0]}"
                     f" Test samples {data[data['is_test'] == True].shape[0]}"
                     )
        return data
    
    def _get_entity_columns_names(self, data: pd.DataFrame) -> List[str]:
        feature_cols = [col for col in data.columns if col not in [self.user_column_name, self.item_column_name, self.label_column_name]]
        logging.info(f"Feature columns: {feature_cols}")
        return feature_cols
    
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
        feature_stats.n_users = data[self.user_column_name].nunique()
        feature_stats.n_items = data[self.item_column_name].nunique()
        return feature_stats
    