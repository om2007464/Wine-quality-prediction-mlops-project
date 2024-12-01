import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import DataIngestionConfig
import mlflow
from comet_ml import Experiment
from prefect import task, Flow
import pymongo
from pymongo import MongoClient
from datetime import datetime

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.client = MongoClient('mongodb://localhost:27017/')  # Connect to MongoDB (adjust URL if needed)
        self.db = self.client['wine_quality']  # Choose the database
        self.collection = self.db['data_ingestion']  # Choose the collection

    @task
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
            # Log download details to MongoDB
            self.log_to_mongo({
                "file_name": filename,
                "source_URL": self.config.source_URL,
                "status": "downloaded",
                "timestamp": datetime.now()
            })
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    @task
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Data extracted to {unzip_path}")
        # Log extraction details to MongoDB
        self.log_to_mongo({
            "file_name": self.config.local_data_file,
            "unzip_path": unzip_path,
            "status": "extracted",
            "timestamp": datetime.now()
        })

    def log_to_mongo(self, log_data):
        """Helper function to log data to MongoDB."""
        self.collection.insert_one(log_data)
        logger.info(f"Logged data to MongoDB: {log_data}")

    def run_prefect_flow(self):
        with Flow("Data Ingestion Flow") as flow:
            download_task = self.download_file()
            extract_task = self.extract_zip_file(upstream_tasks=[download_task])
        flow.run()

    def log_into_mlflow(self):
        mlflow.start_run()
        mlflow.log_param("source_URL", self.config.source_URL)
        mlflow.log_param("local_data_file", self.config.local_data_file)
        mlflow.end_run()

        # Initialize Comet ML experiment
        experiment = Experiment(api_key="Add your API key here", project_name="add your project name here")
        experiment.log_parameters({"source_URL": self.config.source_URL, "local_data_file": self.config.local_data_file})
