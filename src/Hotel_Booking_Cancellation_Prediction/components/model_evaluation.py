import pandas as pd
import joblib
import os
import numpy as np
from Hotel_Booking_Cancellation_Prediction.logging import logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss
)

# components

class ModelEvaluation:

    def __init__(self, config):

        self.config = config

    # LOAD MODEL

    def load_model(self):

        model = joblib.load(self.config.model_path)

        return model

    # LOAD TEST DATA

    def load_data(self):

        test_df = pd.read_csv(self.config.test_data_path)

        return test_df

    # EVALUATE MODEL

    def evaluate_model(self):

        # Load model

        model = self.load_model()

        # Load test data

        test_df = self.load_data()

        # Separate features and target

        X_test = test_df.drop(columns=["is_canceled"])

        y_test = test_df["is_canceled"]

        # Predictions

        y_pred = model.predict(X_test)

        # Probability / decision scores

        if hasattr(model,"predict_proba"):

            y_prob = model.predict_proba(X_test)[:, 1]

        else:

            y_score = model.decision_function(X_test)

            y_prob = (
                1 /
                (
                    1 +
                    __import__("numpy").exp(
                        -y_score
                    )
                )
            )

        # Confusion Matrix

        tn, fp, fn, tp = confusion_matrix(y_test,y_pred
        ).ravel()

        # Accuracy

        accuracy = accuracy_score(y_test,y_pred)

        # Precision

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # Recall

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # Specificity

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        # F1 Score

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # ROC-AUC

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # PR-AUC

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        # Log Loss

        logloss = log_loss(
            y_test,
            y_prob
        )

        # Create results

        results = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "Specificity": specificity,

            "F1 Score": f1,

            "ROC-AUC": roc_auc,

            "PR-AUC": pr_auc,

            "Log Loss": logloss
        }

        # Display results

        print("\nFinal Model Evaluation:")

        for metric, value in results.items():

            print(f"{metric}: {value:.4f}")

        # Save results

        results_df = pd.DataFrame([results])

        results_df.to_csv(
            self.config.metric_file_name,
            index=False
        )

        print( "\nEvaluation metrics saved at:")

        print(self.config.metric_file_name)

        return results_df