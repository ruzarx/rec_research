
from typing import Tuple
from data_preparation.steam_games.data_processor_clean_interactions import SteamDataProcessorClean
from data_preparation.netflix_movies.data_processor_clean_interactions import NetflixDataProcessorClean
from helper_modules.dataset_class import FeaturesCounts
from torch.utils.data import DataLoader

datasets = {'steam': SteamDataProcessorClean,
            'netflix': NetflixDataProcessorClean,
            'amazon_books': 'amazon_books',
            'movie_lens': 'movie_lens_32M'}

class DatasetSelector:
    def __init__(self):
        pass

    def get_data(self,
                 dataset_name: str,
                 batch_size: int,
                 min_user_interactions: int,
                 min_item_interactions: int,
                 n_unique_users: int,
                 n_valid_sample: int,
                 ) -> Tuple[DataLoader, DataLoader, FeaturesCounts]:
        if dataset_name in datasets:
            data_processor = datasets[dataset_name](batch_size,
                                                    min_user_interactions,
                                                    min_item_interactions,
                                                    n_valid_sample)
            train_dataloader, valid_dataloader, test_dataloader, features_stats = data_processor.get_dataset(n_unique_users)
        return train_dataloader, valid_dataloader, test_dataloader, features_stats
