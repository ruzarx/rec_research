

import torch
from data_selector import DatasetSelector
from models.matrix_factorisation import MatrixFactorization
from helper_modules.model_trainer import ModelTrainer


data_processing_params = {'batch_size': 512,
          'min_user_interactions': 12,
          'min_game_interactions': 7,
          'neg_sample_multiplier': 20,
          'is_add_features': False}

model_params = {'model_size': 34}

optimizer_params = {'lr': 2.56e-05}
          
training_params = {'epochs': 16}

def prepare_dataset(dataset_name: str):
    dataset_selector = DatasetSelector()
    (train_dataloader,
    valid_dataloader,
    feature_stats) = dataset_selector.get_data(dataset_name, **data_processing_params)
    return train_dataloader, valid_dataloader, feature_stats

train_dataloader, valid_dataloader, feature_stats = prepare_dataset('steam')

device = torch.device('cpu')

model = MatrixFactorization(model_params['model_size'], feature_stats.n_users, feature_stats.n_items)

# loss_function = torch.nn.BCELoss()
pos_weight = torch.tensor([1.588])  # E.g., class_weight > 1 favors the positive class
loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['lr'])


model_trainer = ModelTrainer(model, device, ['roc_auc', 'f1', 'precision', 'recall'])
model_trainer.train_model(optimizer, loss_function, train_dataloader, valid_dataloader, training_params['epochs'])
