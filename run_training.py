

import torch
import numpy as np
from data_selector import DatasetSelector
from helper_modules.model_trainer_bpr import ModelTrainer
from models.matrix_factorisation import MatrixFactorizationBPR, bpr_loss


data_processing_params = {'batch_size': 16,
          'min_user_interactions': 10,
          'min_game_interactions': 14}
model_params = {'model_size': 17}
optimizer_params = {'lr': 0.12}          
training_params = {'epochs': 14}

device = torch.device('cpu')

def prepare_dataset(dataset_name: str, batch_size: int, min_user_interactions: int, min_game_interactions: int):
    dataset_selector = DatasetSelector()
    (train_dataloader,
    valid_dataloader,
    test_dataloader,
    feature_stats) = dataset_selector.get_data(dataset_name, batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions)
    return train_dataloader, valid_dataloader, test_dataloader, feature_stats

train_dataloader, valid_dataloader, test_dataloader, feature_stats = prepare_dataset('steam',
                                                    model_params['model_size'],
                                                    data_processing_params['min_user_interactions'],
                                                    data_processing_params['min_game_interactions'])

def run_training():
    model = MatrixFactorizationBPR(model_params['model_size'], feature_stats.n_users, feature_stats.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['lr'])
    model_trainer = ModelTrainer(model, device)
    loss_function = bpr_loss
    model_trainer.train_model(optimizer, loss_function, train_dataloader, valid_dataloader, num_epochs=training_params['epochs'])
    ndcg_5, ndcg_10 = model_trainer.validate(test_dataloader)
    return ndcg_5, ndcg_10

if __name__ == '__main__':
    metrics_ndcg_5 = []
    metrics_ndcg_10 = []
    for i in range(50):
        print(f"Run {i + 1}")
        ndcg_5, ndcg_10 = run_training()
        metrics_ndcg_5.append(ndcg_5)
        metrics_ndcg_10.append(ndcg_10)

    metrics_ndcg_5 = np.array(metrics_ndcg_5)
    metrics_ndcg_10 = np.array(metrics_ndcg_10)
    
    print('\n\n\n')
    print(f"Average NDCG@5 {metrics_ndcg_5.mean():.5f}, STD {metrics_ndcg_5.std()}")
    print(f"Average NDCG@10 {metrics_ndcg_10.mean():.5f}, STD {metrics_ndcg_10.std()}")
