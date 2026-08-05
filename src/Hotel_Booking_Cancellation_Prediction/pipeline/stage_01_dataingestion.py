from Hotel_Booking_Cancellation_Prediction.config.configuration import ConfigurationManager
from Hotel_Booking_Cancellation_Prediction.components.data_ingestion import DataIngestion
from Hotel_Booking_Cancellation_Prediction.logging import logger


import zipfile
import os

class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>>> Data Ingestion Stage Started <<<<<<")

            config = ConfigurationManager()

            data_ingestion_config = config.get_data_ingestion_config()

            data_ingestion = DataIngestion(config=data_ingestion_config)

            data_ingestion.download_files()

            data_ingestion.extract_zip_file()

            logger.info(">>>>>> Data Ingestion Stage Completed <<<<<<")

        except Exception as e:
            logger.exception(e)
            raise e