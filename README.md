## 🍷 Wine Quality Prediction - MLOps Project

### Overview

This project aims to predict the quality of wine based on various chemical properties and features, such as acidity, alcohol content, pH, and more. The goal is to build a machine learning model that can accurately predict wine quality based on these features, which can be useful for wine manufacturers and researchers to improve quality control.

The project is designed with an **MLOps** approach to ensure smooth deployment, model monitoring, and continuous integration and delivery (CI/CD). It uses a variety of tools and technologies to manage the entire machine learning lifecycle, from data preprocessing to model deployment and monitoring in production.

### Key Features
- **Data Preprocessing & Feature Engineering**: The dataset is cleaned and transformed to create meaningful features for the machine learning model.
- **Model Training & Evaluation**: Several machine learning algorithms, such as XGBoost, are used to train the model, and its performance is evaluated using various metrics.
- **Model Deployment**: The trained model is deployed using Flask, where users can input wine features and get predictions on wine quality.
- **MLOps Workflow**: The project integrates tools like Prefect, Kubeflow, and GitHub Actions to automate workflows, track experiments, and monitor models in production.
- **Model Monitoring**: The performance of the deployed model is continuously monitored using tools like Evidently AI and WhyLabs to detect any potential issues like model drift.

## 🍷 Wine Quality Prediction Tech Stack 🛠️

| **Category**              | **Tools/Technologies**          | **Description**                                                                 |
|---------------------------|---------------------------------|---------------------------------------------------------------------------------|
| **Frontend**               | HTML                            | This project currently doesn't include a frontend for real-time UI predictions. |
| **Backend**                | Flask                           | Web framework used to build a simple API for model inference and predictions.   |
| **Modeling**               | XGBoost, Scikit-Learn, Python   | Machine learning models for predicting wine quality based on input features.    |
| **Database**               | MongoDB                         | NoSQL database used for storing processed data, feature sets, and predictions.  |
| **Orchestration**          | Prefect                         | Orchestrates workflows like data ingestion, training, and model predictions.    |
| **Experiment Tracking**    | MLflow, Comet ML                | Tools for tracking experiments, model parameters, metrics, and version control. |
| **CI/CD**                  | GitHub Actions                  | Automates CI/CD pipelines for model testing, building, and deployment.         |
| **Containerization**       | Docker                          | Containerizes the environment for consistent deployment and reproducibility.   |
| **Model Monitoring**       | Evidently AI                    | Monitors model performance in production to detect drift and degradation.      |
| **Cloud Storage**          | AWS S3                          | Cloud storage for datasets, model artifacts, and logs.                         |
| **Cloud Hosting**          | AWS EC2                         | Cloud hosting for running the API and serving predictions.                     |




## 📊 Dataset and Features

### Dataset

The dataset used in this project is the **Wine Quality Dataset**, which contains samples of red wine and includes various physicochemical properties of the wines. The goal of this dataset is to predict the quality of wine based on these features.

The dataset is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

- **Number of Samples**: 1599 (Red wines)
- **Number of Features**: 12
- **Target Variable**: `quality` (Wine quality, rated from 0 to 10)

### Features

The dataset contains the following **features**:

1. **Fixed Acidity**: Acid content in wine, measured as grams of tartaric acid per liter.
2. **Volatile Acidity**: The amount of acetic acid in wine, expressed in grams per liter. Higher levels indicate vinegar-like characteristics.
3. **Citric Acid**: Provides a fresh, fruity taste, measured in grams per liter.
4. **Residual Sugar**: The amount of sugar remaining in the wine after fermentation, measured in grams per liter.
5. **Chlorides**: The salt content in wine, measured in grams per liter.
6. **Free Sulfur Dioxide**: The amount of free sulfur dioxide in wine, which helps preserve the wine, measured in milligrams per liter.
7. **Total Sulfur Dioxide**: The total amount of sulfur dioxide in the wine, both free and bound, measured in milligrams per liter.
8. **Density**: The density of the wine, which is related to alcohol content and sugar levels.
9. **pH**: The acidity or alkalinity of the wine, measured on a scale from 0 to 14.
10. **Sulphates**: The level of sulphates in wine, which act as preservatives and antioxidants.
11. **Alcohol**: The alcohol content of the wine, measured as a percentage.
12. **Quality**: The target variable, indicating the quality of the wine, rated on a scale from 0 to 10. This is the variable that the model predicts.

### Data Preprocessing

In this project, data preprocessing steps like missing value handling, normalization, feature scaling, and encoding categorical variables (if needed) are applied to prepare the data for model training. The features are carefully selected to ensure the best predictive performance for wine quality prediction.


## 🏗️ Architecture

The architecture of this **Wine Quality Prediction** project is designed to streamline the end-to-end machine learning pipeline, from data ingestion to model deployment, monitoring, and continuous integration. It integrates several powerful tools to automate processes, track experiments, and ensure smooth operation.

### High-Level Architecture

```plaintext
+------------------------+       +----------------------------+       +--------------------------+
|  Data Sources          |       |  Data Preprocessing        |       |  Model Training &        |
|  (Wine Dataset)        |-----> |  (Cleaning, Scaling,       |-----> |  Evaluation              |
|                        |       |   Feature Engineering)     |       |  (XGBoost, Model Eval)  |
+------------------------+       +----------------------------+       +--------------------------+
                                                               |
                                                               v
                                                     +----------------------------+
                                                     |  Model Tracking &          |
                                                     |  Experiment Management     |
                                                     |  (MLflow)                  |
                                                     +----------------------------+
                                                               |
                                                               v
                                                     +----------------------------+
                                                     |  Model Deployment          |
                                                     |  (Flask API, Prediction)   |
                                                     +----------------------------+
                                                               |
                                                               v
                                                     +----------------------------+
                                                     |  Model Monitoring          |
                                                     |  (Evidently AI)            |
                                                     +----------------------------+
                                                               |
                                                               v
                                                     +----------------------------+
                                                     |  CI/CD & Automation        |
                                                     |  (GitHub Actions, Docker,  |
                                                     |   Prefect)                 |
                                                     +----------------------------+

```


## 🛠️ Setup Instructions

Follow these steps to set up and run the **Wine Quality Prediction** project on your local machine.

### Prerequisites

Before you begin, ensure that you have the following tools installed:

- **Git**: For version control.
- **Python 3.7+**: Ensure that Python is installed on your machine.
- **Docker** (optional): For containerization and running the app in isolated environments.
- **AWS Account** (optional): For cloud hosting and storage (S3, EC2, etc.).
- **MongoDB**: For storing metadata and logs (can be set up locally or use a managed service).
- **Streamlit**: If you want to run the frontend interface.
- **GitHub Account**: For accessing the project repository and pushing changes.

### Step 1: Clone the Repository

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/om2007464/Wine-quality-prediction-mlops-project.git
cd Wine-quality-prediction-mlops-project
```

Create Virtual Enviroment 

```
python3 -m venv myenv
source myenv/bin/activate

```
 Install Dependencies

 ```
pip install -r requirements.txt

```
Steps for setup 

- **Step 1**: Clone the repository.
- **Step 2**: Set up a virtual environment.
- **Step 3**: Install project dependencies.
- **Step 4**: Optionally, set up MongoDB and Docker.
- **Step 5**: Set up AWS services for cloud deployment.
- **Step 6**: Run the Flask API locally and use Streamlit if needed.
- **Step 7**: Train the model.
- **Step 8**: Set up monitoring.
- **Step 9**: Use CI/CD with GitHub Actions.


