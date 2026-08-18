
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

# from config.ymal

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen = True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    unzip_data_dir: Path
    all_schema: dict    

@dataclass(frozen=True)
class DataTransformationConfig:

    root_dir: Path
    data_path: Path
    transformed_train_path: Path  # scaled train path # scaled means changing the data to numeric.
    transformed_test_path: Path   # scaled train path # scaled data is used for logistic,naviebayes,svm,KNN
    transformed_unscaled_train_path: Path  # unscaled data is usful for decision tree and random forest
    transformed_unscaled_test_path: Path
    preprocessor_scaled_path: Path
    preprocessor_unscaled_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:

    root_dir: Path
    train_data_path: Path  
    test_data_path: Path
    trained_model_path: Path
    model_report_path: Path     


@dataclass(frozen=True)
class ModelEvaluationConfig:

    root_dir: Path
    model_path: Path
    test_data_path: Path
    metric_file_name: Path    

@dataclass(frozen=True)
class PredictionPipelineConfig:

    model_path: Path
    preprocessor_unscaled_path: Path    