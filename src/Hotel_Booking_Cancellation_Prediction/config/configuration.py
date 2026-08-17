from Hotel_Booking_Cancellation_Prediction.logging import logger
from Hotel_Booking_Cancellation_Prediction.constant import *
from pathlib import Path
from Hotel_Booking_Cancellation_Prediction.utils.common import read_yaml, create_directories
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import DataIngestionConfig
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import DataValidationConfig
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import DataTransformationConfig
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import ModelTrainerConfig
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        schema = read_yaml(Path(config.all_schema)) ## here temp because i didnt write 32 colums in config.yaml so i mentioned temp for schema checking

        data_validation_config = DataValidationConfig(

            root_dir=config.root_dir,

            STATUS_FILE=config.STATUS_FILE,

            unzip_data_dir=config.unzip_data_dir,

            all_schema=schema.COLUMNS
        )

        return data_validation_config


    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(

            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformed_train_path=Path(config.transformed_train_path),
            transformed_test_path=Path(config.transformed_test_path),
            transformed_unscaled_train_path=Path(config.transformed_unscaled_train_path),
            transformed_unscaled_test_path=Path(config.transformed_unscaled_test_path),
            preprocessor_scaled_path=Path(config.preprocessor_scaled_path),
            preprocessor_unscaled_path=Path(config.preprocessor_unscaled_path)
        )

        return data_transformation_config 


    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config["model_trainer"]

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(

            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            trained_model_path=Path(config.trained_model_path),
            model_report_path=Path(config.model_report_path)
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config["model_evaluation"]
 
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(

            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            test_data_path=Path(config.test_data_path),
            metric_file_name=Path(config.metric_file_name)
        )

        return model_evaluation_config