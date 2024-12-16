

import torch
import optuna
from data_selector import DatasetSelector
from helper_modules.model_trainer_bpr import ModelTrainer
from models.matrix_factorisation import MatrixFactorizationBPR, bpr_loss

'''
Best is trial 41 with value: 0.5779737234115601.
Best hyperparameters: {'model_size': 1, 'lr': 0.0020176571386976328, 'min_user_interactions': 6, 'min_game_interactions': 10, 'batch_size': 64}
'''

device = torch.device('cpu')

data_processing_params = {'batch_size': 128,
          'min_user_interactions': 50,
          'min_item_interactions': 50,
          'n_unique_users': 50_000,
          'n_valid_sample': 10}
model_params = {'model_size': 4}
optimizer_params = {'lr': 0.005}
training_params = {'epochs': 20}

def prepare_dataset(dataset_name: str,
                    batch_size: int,
                    min_user_interactions: int,
                    min_item_interactions: int,
                    n_unique_users: int,
                    n_valid_sample: int):
    dataset_selector = DatasetSelector()
    (train_dataloader,
    valid_dataloader,
    test_dataloader,
    feature_stats) = dataset_selector.get_data(dataset_name,
                                               batch_size,
                                                min_user_interactions,
                                                min_item_interactions,
                                                n_unique_users,
                                                n_valid_sample)
    return train_dataloader, valid_dataloader, test_dataloader, feature_stats

train_dataloader, valid_dataloader, _, feature_stats = prepare_dataset('netflix',
                                                data_processing_params['batch_size'],
                                                data_processing_params['min_user_interactions'],
                                                data_processing_params['min_item_interactions'],
                                                data_processing_params['n_unique_users'],
                                                data_processing_params['n_valid_sample'])

def objective(trial):
    model_size = trial.suggest_int("model_size", 1, 64)  # Tuning model size
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # Tuning learning rate
    # min_user_interactions = trial.suggest_int("min_user_interactions", 30, 120)
    # min_item_interactions = trial.suggest_int("min_item_interactions", 30, 120)
    # batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    model = MatrixFactorizationBPR(model_size, feature_stats.n_users, feature_stats.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_trainer = ModelTrainer(model, device)
    loss_function = bpr_loss
    ndcg_5 = model_trainer.train_model(optimizer, loss_function, train_dataloader, valid_dataloader, num_epochs=training_params['epochs'])
    # ndcg_5, ndcg_10 = model_trainer.validate(test_dataloader)
    return ndcg_5


def validate_best_params(best_params):
    model_size = best_params["model_size"]
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]
    min_user_interactions = best_params["min_user_interactions"]
    min_item_interactions = best_params["min_item_interactions"]
    train_dataloader, _, test_dataloader, feature_stats = prepare_dataset('netflix',
                                                    batch_size,
                                                    min_user_interactions,
                                                    min_item_interactions,
                                                    data_processing_params['n_unique_users'],
                                                    data_processing_params['n_valid_sample'])
    model = MatrixFactorizationBPR(model_size, feature_stats.n_users, feature_stats.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_trainer = ModelTrainer(model, device)
    model_trainer.train_model(optimizer, train_dataloader, test_dataloader, num_epochs=20)
    ndcg_5, ndcg_10 = model_trainer.validate(test_dataloader)
    print(f"Best model validation NDCG@5 {ndcg_5:.3f}, NDCG@10 {ndcg_10:.3f}")
    return

# Run Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"Validation of model with the best params")
    print(best_params)
    validate_best_params(best_params)

    # Print the best hyperparameters
    print("Best hyperparameters:", best_params)
