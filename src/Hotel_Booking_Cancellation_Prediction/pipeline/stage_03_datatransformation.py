from Hotel_Booking_Cancellation_Prediction.config.configuration import ConfigurationManager
from Hotel_Booking_Cancellation_Prediction.components.data_transformation import DataTransformation
from Hotel_Booking_Cancellation_Prediction.logging import logger


class DataTransformationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:

            logger.info(">>>>>> Data Transformation Stage Started <<<<<<")

            config = ConfigurationManager()

            data_transformation_config = (config.get_data_transformation_config() )

            data_transformation = DataTransformation(config=data_transformation_config)

            data_transformation.transform_data()

            logger.info(">>>>>> Data Transformation Stage Completed <<<<<<")
            

        except Exception as e:

            logger.exception(e)
            raise e