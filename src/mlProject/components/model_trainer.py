import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import joblib
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import ModelTrainerConfig
import optuna
from prefect import task

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def optimize_params(self):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.01, 1.0)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            return model

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        return best_params

    @task
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        best_params = self.optimize_params()
        model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'], random_state=42)
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

        


