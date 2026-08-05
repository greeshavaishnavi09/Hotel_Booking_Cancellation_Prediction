from pathlib import Path
import yaml
import sys
import os
import json
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from ensure import ensure_annotations

from Hotel_Booking_Cancellation_Prediction.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox object.

    Args:
        path_to_yaml (Path): Path to YAML file

    Returns:
        ConfigBox: Parsed YAML content
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """create list of directories
    
    args:
        path to directories (list) : list of path to directories
        ignore_log (bool, optional): ignore if multiple directories is to be created. default to false
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb

    args:
        path (Path): path of the file
        
    Return:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def save_json(path: Path, data: dict):
    """saves a dictionary as a json file

    Args:
        path (Path): path to the destination json file
        data (dict): dictionary data to be saved
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved successfully at: {path}")