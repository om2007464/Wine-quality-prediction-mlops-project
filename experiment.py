import mlflow
import mlflow.sklearn
import logging as logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from mlProject.entity.config_entity import DataValidationConfig
from mlProject.components.data_validation import DataValidation
from prefect import task, flow

# Setup logger
logger.basicConfig(level=logger.INFO)
logger = logger.getLogger("mlProject")

# MLflow Experiment and Flow
mlflow.set_experiment('wine_quality_experiment')

# Prefect Flow to handle experiments
@flow
def machine_learning_pipeline(config: DataValidationConfig):
    # Step 1: Data Validation
    data_validation = DataValidation(config)
    validation_status = data_validation.validate_all_columns()
    
    if not validation_status:
        logger.error("Data validation failed. Exiting pipeline.")
        return
    
    # Step 2: Model Training and Logging with MLflow
    data = pd.read_csv(config.unzip_data_dir)
    target_column = "quality"
    
    # Step 3: Run with RandomForest and Logistic Regression
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "LogisticRegression": LogisticRegression(max_iter=100)
    }
    
    for model_name, model in models.items():
        run_model(data, model, model_name, target_column)

import time

@task
def run_model(data: pd.DataFrame, model, model_name: str, target_column: str):
    start_time = time.time()
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=target_column), data[target_column], test_size=0.3, random_state=42)
        
        # Start MLflow Run
        with mlflow.start_run():
            # Train model
            model.fit(X_train, y_train)
            
            # Log hyperparameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_estimators" if model_name == "RandomForest" else "max_iter", model.n_estimators if model_name == "RandomForest" else model.max_iter)
            
            # Evaluate model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model: {model_name} - Accuracy: {accuracy}")
    
    except Exception as e:
        logger.error(f"Error during model {model_name} training: {e}")
        raise e
    end_time = time.time()
    logger.info(f"Time taken for {model_name}: {end_time - start_time} seconds")


@task
def run_model(data: pd.DataFrame, model, model_name: str, target_column: str):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=target_column), data[target_column], test_size=0.3, random_state=42)
        
        # Start MLflow Run
        with mlflow.start_run():
            # Train model
            model.fit(X_train, y_train)
            
            # Log hyperparameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_estimators" if model_name == "RandomForest" else "max_iter", model.n_estimators if model_name == "RandomForest" else model.max_iter)
            
            # Evaluate model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model: {model_name} - Accuracy: {accuracy}")
    
    except Exception as e:
        logger.error(f"Error during model {model_name} training: {e}")
        raise e


if __name__ == "__main__":
    # Define configuration (replace with actual paths)
    config = DataValidationConfig(
        unzip_data_dir="path_to_your_dataset.csv",  # Update path
        all_schema = {
            "fixed acidity": "float64",
            "volatile acidity": "float64",
            "citric acid": "float64",
            "residual sugar": "float64",
            "chlorides": "float64",
            "free sulfur dioxide": "float64",
            "total sulfur dioxide": "float64",
            "density": "float64",
            "pH": "float64",
            "sulphates": "float64",
            "alcohol": "float64",
            "quality": "int64"
        },
        STATUS_FILE="status_file",
        root_dir="path_to_root_dir"  # Update root directory
    )
    
    # Run the experiment pipeline
    machine_learning_pipeline(config)
