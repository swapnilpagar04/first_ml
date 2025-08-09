import os
import sys
from src.exeption import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, 'artifacts')
    train_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'test.csv')
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'data.csv')
    source_data_path: str = os.path.join(PROJECT_ROOT, 'notebook', 'data', 'stud.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            # Ensure artifacts directory exists before doing anything
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            logging.info(f"Artifacts directory ready at: {self.ingestion_config.artifacts_dir}")

            # Check if source CSV exists
            if not os.path.exists(self.ingestion_config.source_data_path):
                raise FileNotFoundError(f"Source file not found: {self.ingestion_config.source_data_path}")

            # Read CSV file
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info(f"Dataset loaded from: {self.ingestion_config.source_data_path}")

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Split into train & test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at: {self.ingestion_config.test_data_path}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error in Data Ingestion", exc_info=True)
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
