
from typing import Tuple, List, Dict, Optional
import json
import random
import logging
from torch.utils.data import DataLoader
import pandas as pd

from helper_modules.dataset_class import RecDataset, FeaturesCounts


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamDataProcessor:
    def __init__(self,
                 batch_size: int,
                 min_user_interactions: int,
                 min_game_interactions: int,
                 neg_sample_multiplier: int,
                 is_add_features: bool):
        self.data_path = f"data/steam_games/"
        self.data_file_name = "steam-200k.csv"
        self.batch_size = batch_size
        self.user_column_name = 'user'
        self.item_column_name = 'game'
        self.label_column_name = 'hours'
        self.min_user_interactions = min_user_interactions
        self.min_game_interactions = min_game_interactions
        self.neg_sample_multiplier = neg_sample_multiplier
        self.is_add_features = is_add_features
        self.features_stats = FeaturesCounts()
        return
    
    def get_dataset(self) -> Tuple[DataLoader, DataLoader]:
        data, features_stats = self._prepare_raw_data()
        train_df, valid_df = self._prepare_train_valid_data(data)
        train_dataset = RecDataset(train_df,
                                    self.user_column_name,
                                    self.item_column_name,
                                    self.label_column_name,
                                    features_stats.user_features,
                                    features_stats.user_features)
        valid_dataset = RecDataset(valid_df,
                                    self.user_column_name,
                                    self.item_column_name,
                                    self.label_column_name,
                                    features_stats.user_features,
                                    features_stats.user_features)
        train_dataloader = RecDataset.get_data_loader(train_dataset, self.batch_size)
        valid_dataloader = RecDataset.get_data_loader(valid_dataset, self.batch_size)
        logging.info(f"Created dataloaders for train {len(train_dataloader)} and validation {len(valid_dataloader)}")
        return train_dataloader, valid_dataloader, features_stats
    
    def _prepare_raw_data(self) -> Tuple[pd.DataFrame, FeaturesCounts]:
        df = self._load_raw_data()
        df = self._filter_duplicates(df)
        df = self._filter_min_interactions(df)
        df = self._define_validation(df)
        df = self._add_valid_negative_samples(df)
        user_features = self._process_user_features(df)
        game_features = self._process_game_features(df)
        df = self._merge_with_features(df, user_features, game_features)
        user_feature_names = self._get_entity_columns_names(user_features)
        item_feature_names = self._get_entity_columns_names(game_features)
        df = self._keep_purchases_only(df)
        df = self._encode_entity(df, 'user')
        df = self._encode_entity(df, 'game')
        features_stats = self._collect_feature_stats(df, user_feature_names, item_feature_names)
        return df, features_stats
    
    def _prepare_train_valid_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, valid_df = self._split_train_valid(data)
        train_df, valid_df, normalization_values = self._normalize_features(train_df, valid_df)
        logging.info(f"Split data into train {train_df.shape} and validation {valid_df.shape}")
        self._store_encoding_data(normalization_values, 'normalization')
        return train_df, valid_df

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
        # data[entity_column] = data[entity_column].replace(encoding).astype(data[entity_column].dtype)
        data[entity_column] = data[entity_column].map(encoding).fillna(-1)
        logging.info(f"Encoded {entity_column}, {len(encoding)} unique values")
        self._store_encoding_data(rev_encoding, f"{entity_column}_encoding")
        return data
    
    def _process_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        n_games_played_per_user = (
            data[data['action'] == 'play'][['user', 'game', 'hours']]
            .groupby('user', as_index=False)
            .agg(
                games_played_by_user_count=('game', 'count'),
                hours_played_by_user_sum=('hours', 'sum'),
                hours_played_by_user_mean=('hours', 'mean'),
                hours_played_by_user_min=('hours', 'min'),
                hours_played_by_user_max=('hours', 'max'),
                hours_played_by_user_median=('hours', 'median'),
            )
        )
        n_games_bought_per_user = (
            data[data['action'] == 'purchase'][['user', 'game']]
            .groupby('user', as_index=False)
            .agg(
                games_bought_by_user_count=('game', 'count'),
            )
        )
        user_features = n_games_bought_per_user.merge(n_games_played_per_user, on='user', how='left').fillna(0)
        user_features['games_played_to_bought_by_user_ratio'] = (
            user_features['games_played_by_user_count'] / 
            user_features['games_bought_by_user_count']
        )
        logging.info(f"Added user features, overall shape {user_features.shape}")
        return user_features
    
    def _process_game_features(self, data: pd.DataFrame) -> pd.DataFrame:
        n_users_played_per_user = (
            data[data['action'] == 'play'][['user', 'game', 'hours']]
            .groupby('game', as_index=False)
            .agg(
                game_played_count=('user', 'count'),
                game_hours_sum=('hours', 'sum'),
                game_hours_mean=('hours', 'mean'),
                game_hours_min=('hours', 'min'),
                game_hours_max=('hours', 'max'),
                game_hours_median=('hours', 'median'),
            )
        )
        n_users_bought_per_user = (
            data[data['action'] == 'purchase'][['user', 'game']]
            .groupby('game', as_index=False)
            .agg(
                game_bought_count=('user', 'count'),
            )
        )
        game_features = n_users_bought_per_user.merge(n_users_played_per_user, on='game', how='left').fillna(0)
        game_features['game_played_to_bought_ratio'] = (
            game_features['game_played_count'] / 
            game_features['game_bought_count']
        )
        logging.info(f"Added games features, overall shape {game_features.shape}")
        return game_features
    
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
        data.loc[
            data[data['action'] == 'purchase'].sort_values(['user', 'n_event']).groupby('user').tail(3).index,
            'is_valid'] = True
        data = data.drop(columns='n_event')
        logging.info(f"Added validation colums. "
                     f" Validation samples {data[data['is_valid'] == True].shape[0]}"
                     f" Train samples {data[data['is_valid'] == False].shape[0]}"
                     )
        return data

    def _add_valid_negative_samples(self, data: pd.DataFrame) -> pd.DataFrame:
        user_games = dict()
        for _, user, game in data[['user', 'game']].itertuples():
            if user in user_games:
                user_games[user].add(game)
            else:
                user_games[user] = set([game])

        all_games = data['game'].unique()
        val_pair_users = []
        val_pair_games = []
        for user, games in user_games.items():
            for _ in range(int(self.neg_sample_multiplier)):
                game_candidate = random.choice(all_games)
                while game_candidate in games:
                    game_candidate = random.choice(all_games)
                val_pair_users.append(user)
                val_pair_games.append(game_candidate)

        negative_val_sample = pd.DataFrame({'user': val_pair_users, 'game': val_pair_games})
        negative_val_sample['is_valid'] = True
        negative_val_sample['action'] = 'purchase'
        negative_val_sample['hours'] = 0
        data = pd.concat([data, negative_val_sample])
        logging.info(f"Added negative samples {self.neg_sample_multiplier} per user. Resulting shape {data.shape[0]}")
        logging.info(f"Validation proportion: 1 - {data[(data['is_valid'] == True) & (data['hours'] == 1)].shape[0]}, \
                      1 - {data[(data['is_valid'] == True) & (data['hours'] == 0)].shape[0]}")
        return data
    
    def _get_entity_columns_names(self, data: pd.DataFrame) -> List[str]:
        feature_cols = [col for col in data.columns if col not in ['user', 'game', 'hours']]
        logging.info(f"Feature columns: {feature_cols}")
        return feature_cols
    
    def _merge_with_features(self,
                             data: pd.DataFrame,
                             user_features: pd.DataFrame,
                             game_features: pd.DataFrame
                             ) -> pd.DataFrame:
        data = data.merge(user_features, on='user', how='left')
        data = data.merge(game_features, on='game', how='left')
        logging.info(f"Merged with features: {data.shape}")
        return data
    
    def _keep_purchases_only(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[data['action'] == 'purchase'].reset_index(drop=True)
        logging.info(f"Dropped play data: {data.shape}")
        return data
    
    def _split_train_valid(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = data[data['is_valid'] == False].reset_index(drop=True).drop(columns='is_valid')
        valid_df = data[data['is_valid'] == True].reset_index(drop=True).drop(columns='is_valid')
        return train_df, valid_df

    def _normalize_features(self,
                            train_df: pd.DataFrame,
                            valid_df: pd.DataFrame
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
        features_to_std = [col for col in train_df.columns if col not in ['user', 'game', 'hours', 'action', 'is_valid']]
        normalization_values = dict()
        for col in features_to_std:
            if col in train_df.columns:
                std = train_df[col].std()
                mean = train_df[col].mean()
                train_df[col] = (train_df[col] - mean) / std
                valid_df[col] = (valid_df[col] - mean) / std
                normalization_values[col] = (std, mean)
        return train_df, valid_df, normalization_values
    
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
    