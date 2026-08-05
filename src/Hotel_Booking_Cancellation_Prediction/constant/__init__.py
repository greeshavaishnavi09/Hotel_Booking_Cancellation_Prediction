from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
SCHEMA_FILE_PATH = ROOT_DIR / "schema.yaml"

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"