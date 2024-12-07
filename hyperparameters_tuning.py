

import torch
import optuna
from data_selector import DatasetSelector
from models.matrix_factorisation import MatrixFactorization
from helper_modules.model_trainer import ModelTrainer


data_processing_params = {'batch_size': 256,
          'min_user_interactions': 1,
          'min_game_interactions': 1,
          'neg_sample_multiplier': 1,
          'is_add_features': False}

model_params = {'model_size': 2}

optimizer_params = {'lr': 0.001}
          
training_params = {'epochs': 16}

device = torch.device('cpu')

def prepare_dataset(dataset_name: str, batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions,
                                                    neg_sample_multiplier,
                                                    is_add_features):
    dataset_selector = DatasetSelector()
    (train_dataloader,
    valid_dataloader,
    feature_stats) = dataset_selector.get_data(dataset_name, batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions,
                                                    neg_sample_multiplier,
                                                    is_add_features)
    return train_dataloader, valid_dataloader, feature_stats

def objective(trial):
    model_size = trial.suggest_int("model_size", 1, 50)  # Tuning model size
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-0, log=True)  # Tuning learning rate
    pos_weight_value = trial.suggest_float("pos_weight", 0.5, 2.0)  # Tuning class weight
    min_user_interactions = trial.suggest_int("min_user_interactions", 5, 15)
    min_game_interactions = trial.suggest_int("min_game_interactions", 5, 15)
    batch_size = trial.suggest_int("batch_size", 8, 1024)

    train_dataloader, valid_dataloader, feature_stats = prepare_dataset('steam', batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions,
                                                    20,
                                                    False)

    model = MatrixFactorization(model_size, feature_stats.n_users, feature_stats.n_items).to(device)

    pos_weight = torch.tensor([pos_weight_value], device=device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_trainer = ModelTrainer(model, device, ['roc_auc', 'f1', 'precision', 'recall'])

    precision_at_k = model_trainer.train_model(optimizer, loss_function, train_dataloader, valid_dataloader, 50)
    # metrics = model_trainer.metrics

    # for metric in metrics:
        # if metric.name == 'F1 Score':
            # return metric.value
    return precision_at_k

    # # Objective is to maximize ROC-AUC
    # roc_auc = metrics['roc_auc']
    # return roc_auc


# Run Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
