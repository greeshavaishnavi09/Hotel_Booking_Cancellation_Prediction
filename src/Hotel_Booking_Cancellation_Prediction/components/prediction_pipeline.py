# components

import pandas as pd
import joblib


class PredictionPipeline:

    def __init__(self, config):

        self.config = config

    # LOAD MODEL

    def load_model(self):

        model = joblib.load(self.config.model_path)

        return model

    # LOAD PREPROCESSOR

    def load_preprocessor(self):

        preprocessor = joblib.load(
            self.config.preprocessor_unscaled_path
        )

        return preprocessor

    # MAKE PREDICTION

    def predict(self, input_data):

        # Load trained model

        model = self.load_model()

        # Load saved preprocessor

        preprocessor = self.load_preprocessor()

        # Convert user input into DataFrame

        input_df = pd.DataFrame([input_data])

        # Apply the SAME preprocessing used during model training i.e., unscaled data

        transformed_input = (
            preprocessor.transform(
                input_df
            )
        )

        # Make prediction

        prediction = model.predict(
            transformed_input)[0]

        # Get prediction probability

        if hasattr(
            model,
            "predict_proba"):

            probability = model.predict_proba(
                transformed_input
            )[0][1]

        else:

            probability = None

        return prediction, probability