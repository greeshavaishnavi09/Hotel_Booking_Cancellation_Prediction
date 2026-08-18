from Hotel_Booking_Cancellation_Prediction.config.configuration import ConfigurationManager
from Hotel_Booking_Cancellation_Prediction.components.model_evaluation import ModelEvaluation
from Hotel_Booking_Cancellation_Prediction.logging import logger


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:

            logger.info(">>>>>> Model Evaluation Stage Started <<<<<<")

            config = ConfigurationManager()

            model_evaluation_config = (config.get_model_evaluation_config() )

            model_evaluation = ModelEvaluation(config=model_evaluation_config)

            model_evaluation.evaluate_model()

            logger.info(">>>>>> Model Evaluation Stage Completed <<<<<<")

        except Exception as e:

            logger.exception(e)
            raise e
