from Hotel_Booking_Cancellation_Prediction.config.configuration import ConfigurationManager
from Hotel_Booking_Cancellation_Prediction.components.prediction_pipeline import PredictionPipeline
from Hotel_Booking_Cancellation_Prediction.logging import logger

# pipeline

class PredictionPipelineTrainingPipeline:

    def __init__(self):
        pass

    def main(self,input_data):

        try:

            logger.info(">>>>>> Prediction Pipeline Stage Started <<<<<<")

            config = ConfigurationManager()

            prediction_pipeline_config = (config.get_prediction_pipeline_config() )

            prediction_pipeline = PredictionPipeline(config=prediction_pipeline_config)

            prediction, probability = (prediction_pipeline.predict(input_data))

            logger.info(">>>>>> Prediction Pipeline Stage Completed <<<<<<")

            return prediction, probability


        except Exception as e:

            logger.exception(e)
            raise e