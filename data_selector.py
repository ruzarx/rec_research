
from typing import Tuple
from data_preparation.steam_games.data_processor_clean_interactions import SteamDataProcessorClean
from helper_modules.dataset_class import FeaturesCounts
from torch.utils.data import DataLoader

datasets = {'steam': SteamDataProcessorClean,
            'netflix': 'netflix_movies',
            'amazon_books': 'amazon_books',
            'movie_lens': 'movie_lens_32M'}

class DatasetSelector:
    def __init__(self):
        pass

    def get_data(self,
                 dataset_name: str,
                 batch_size: int,
                 min_user_interactions: int,
                 min_game_interactions: int,
                 ) -> Tuple[DataLoader, DataLoader, FeaturesCounts]:
        if dataset_name in datasets:
            data_processor = datasets[dataset_name](batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions)
            train_dataloader, valid_dataloader, test_dataloader, features_stats = data_processor.get_dataset()
        return train_dataloader, valid_dataloader, test_dataloader, features_stats
