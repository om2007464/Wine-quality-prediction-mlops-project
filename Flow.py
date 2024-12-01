import pandas as pd
from prefect import flow, task
from mlProject.entity.config_entity import DataValidationConfig
from mlProject.components.data_validation import DataValidation
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import logging

# Setup logger
logger = logging.getLogger("mlProject")

# Prefect flow for the entire machine learning pipeline
@flow
def machine_learning_pipeline(config: DataValidationConfig):
    
    # Step 1: Data Validation
    data_validation = DataValidation(config)
    validation_status = data_validation.validate_all_columns()
    
    if not validation_status:
        logger.error("Data validation failed. Exiting pipeline.")
        return
    
    # Step 2: Model Training
    data = pd.read_csv(config.unzip_data_dir)
    target_column = "quality"
    
    # Train RandomForestClassifier
    model = train_model(data, target_column)
    
    # Step 3: Generate LIME Explanation for a single instance
    generate_lime_explanation(data, model, target_column)

@task
def train_model(data: pd.DataFrame, target_column: str) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier()
        model.fit(data.drop(columns=target_column), data[target_column])
        
        # Get feature importances and plot
        feature_importances = model.feature_importances_
        plot_feature_importance(data, feature_importances, target_column)
        
        logger.info("Model training completed.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise e

@task
def plot_feature_importance(data: pd.DataFrame, feature_importances: list, target_column: str):
    feature_names = data.drop(columns=target_column).columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.show()

@task
def generate_lime_explanation(data: pd.DataFrame, model: RandomForestClassifier, target_column: str):
    try:
        # Prepare the LIME explainer
        lime_explainer = LimeTabularExplainer(
            training_data=data.drop(columns=target_column).values,
            feature_names=data.drop(columns=target_column).columns,
            class_names=[str(cls) for cls in model.classes_],
            mode="classification"
        )

        # Choose a random instance (you can modify this)
        instance_index = 0
        instance = data.drop(columns=target_column).iloc[instance_index].values.reshape(1, -1)

        # Generate LIME explanation
        explanation = lime_explainer.explain_instance(
            data_row=instance[0],
            predict_fn=model.predict_proba
        )

        # Show the explanation in a notebook or console
        explanation.show_in_notebook(show_table=True)
        
        logger.info("LIME explanation generated for the selected instance.")
    except Exception as e:
        logger.error(f"Error during LIME explanation generation: {e}")
        raise e

# Example usage: Update config with actual paths
if __name__ == "__main__":
    # Define your configuration with actual paths
    config = DataValidationConfig(
        unzip_data_dir="artifacts/data_ingestion/data.zip",
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
        STATUS_FILE="status.txt",  # Correct path
        root_dir="artifacts"  # Actual root directory path
    )

    # Run the entire flow
    machine_learning_pipeline(config)
