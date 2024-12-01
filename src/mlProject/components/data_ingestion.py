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

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @task
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    @task
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

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
        experiment = Experiment(api_key="Add your API key here", project_name="add your project name here ")
        experiment.log_parameters({"source_URL": self.config.source_URL, "local_data_file": self.config.local_data_file})
