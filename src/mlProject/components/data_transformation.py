import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
import optuna
from prefect import task


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @task
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)

    def optimize_params(self):
        def objective(trial):
            # Hyperparameter optimization logic here, for example, scaling factors, etc.
            scale_factor = trial.suggest_uniform('scale_factor', 0.1, 1.0)
            # Optimize preprocessing or feature engineering steps based on the trial results
            return scale_factor

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
