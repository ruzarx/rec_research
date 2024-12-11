

import torch
import optuna
from data_selector import DatasetSelector
from helper_modules.model_trainer_bpr import MatrixFactorizationBPR, ModelTrainer

'''
Best is trial 41 with value: 0.5779737234115601.
Best hyperparameters: {'model_size': 1, 'lr': 0.0020176571386976328, 'min_user_interactions': 6, 'min_game_interactions': 10, 'batch_size': 64}
'''

device = torch.device('cpu')

def prepare_dataset(dataset_name: str, batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions,
                                                    neg_sample_multiplier,
                                                    is_add_features):
    dataset_selector = DatasetSelector()
    (train_dataloader,
    valid_dataloader,
    test_dataloader,
    feature_stats) = dataset_selector.get_data(dataset_name, batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions,
                                                    neg_sample_multiplier,
                                                    is_add_features)
    return train_dataloader, valid_dataloader, test_dataloader, feature_stats

def objective(trial):
    model_size = trial.suggest_int("model_size", 1, 20)  # Tuning model size
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)  # Tuning learning rate
    min_user_interactions = trial.suggest_int("min_user_interactions", 8, 20)
    min_game_interactions = trial.suggest_int("min_game_interactions", 8, 20)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    train_dataloader, valid_dataloader, _, feature_stats = prepare_dataset('steam', batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions)

    model = MatrixFactorizationBPR(model_size, feature_stats.n_users, feature_stats.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_trainer = ModelTrainer(model, device)
    ndcg_5, ndcg_10 = model_trainer.train_model(optimizer, train_dataloader, valid_dataloader, num_epochs=20)
    return ndcg_5


def validate_best_params(best_params):
    model_size = best_params["model_size"]
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]
    min_user_interactions = best_params["min_user_interactions"]
    min_game_interactions = best_params["min_game_interactions"]
    train_dataloader, _, test_dataloader, feature_stats = prepare_dataset('steam', batch_size,
                                                    min_user_interactions,
                                                    min_game_interactions)
    model = MatrixFactorizationBPR(model_size, feature_stats.n_users, feature_stats.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_trainer = ModelTrainer(model, device)
    model_trainer.train_model(optimizer, train_dataloader, test_dataloader, num_epochs=20)
    return

# Run Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    best_params = study.best_params
    print(f"Validation of model with the best params")
    print(best_params)
    validate_best_params(best_params)

    # Print the best hyperparameters
    print("Best hyperparameters:", best_params)
