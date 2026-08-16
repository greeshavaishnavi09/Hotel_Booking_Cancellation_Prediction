from Hotel_Booking_Cancellation_Prediction.config.configuration import ConfigurationManager
from Hotel_Booking_Cancellation_Prediction.components.model_trainer import ModelTrainer
from Hotel_Booking_Cancellation_Prediction.logging import logger


# pipeline

class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:

            logger.info(">>>>>> Model Trainer Stage Started <<<<<<")

            config = ConfigurationManager()

            model_trainer_config = (config.get_model_trainer_config() )

            model_trainer = ModelTrainer(config=model_trainer_config)

            model_trainer.train_models()

            logger.info(
                ">>>>>> Model Trainer Stage Completed <<<<<<"
            )

        except Exception as e:

            logger.exception(e)
            raise e