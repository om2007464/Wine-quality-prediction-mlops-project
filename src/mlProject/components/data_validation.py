import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlProject.entity.config_entity import DataValidationConfig
from prefect import task
from lime.lime_tabular import LimeTabularExplainer  # Import LIME
import logging as logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    @task
    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            # Read data
            data = pd.read_csv('artifacts/data_ingestion/winequality-red.csv')
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            # Check if columns match schema
            with open(self.config.STATUS_FILE, 'w') as f:
                for col in all_cols:
                    if col not in all_schema:
                        validation_status = False
                        f.write(f"Column {col} not in schema.\n")
                    else:
                        f.write(f"Column {col} is valid.\n")

            # Target column is 'quality'
            target_column = "quality"  # The actual target column name

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset")

            # Example model for feature importance
            model = RandomForestClassifier()  # Replace with your actual model
            model.fit(data.drop(columns=target_column), data[target_column])  # Fit model on data (replace with your target column)

            # Get feature importances
            feature_importances = model.feature_importances_

            # Plot feature importances
            feature_names = data.drop(columns=target_column).columns
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.show()

            # LIME explanation for a single instance
            lime_explainer = LimeTabularExplainer(
                training_data=data.drop(columns=target_column).values,  # Use features for LIME
                feature_names=feature_names,
                class_names=[str(cls) for cls in model.classes_],  # Target class names
                mode="classification"
            )

            # Choose a random instance from the data for explanation
            instance_index = 0  # You can change this to explain different instances
            instance = data.drop(columns=target_column).iloc[instance_index].values.reshape(1, -1)

            # Generate LIME explanation
            explanation = lime_explainer.explain_instance(
                data_row=instance[0],
                predict_fn=model.predict_proba  # Prediction function from the model
            )

            # Show the explanation
            explanation.show_in_notebook(show_table=True)

            return validation_status
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            raise e
