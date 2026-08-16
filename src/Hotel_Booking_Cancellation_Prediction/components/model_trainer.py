# import all libraries

import os
import pandas as pd
import joblib

from Hotel_Booking_Cancellation_Prediction.logging import logger

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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


class ModelTrainer:

    def __init__(self, config):

        self.config = config

    # LOAD TRANSFORMED DATA

    def load_data(self):

        # Scaled data
        train_scaled = pd.read_csv(
            self.config.train_data_path
        )

        test_scaled = pd.read_csv(
            self.config.test_data_path
        )

        # Unscaled data
        unscaled_train_path = os.path.join(
            os.path.dirname(
                self.config.train_data_path
            ),
            "train_unscaled.csv"
        )

        unscaled_test_path = os.path.join(
            os.path.dirname(
                self.config.test_data_path
            ),
            "test_unscaled.csv"
        )

        train_unscaled = pd.read_csv(
            unscaled_train_path
        )

        test_unscaled = pd.read_csv(
            unscaled_test_path
        )

        return (
            train_scaled,
            test_scaled,
            train_unscaled,
            test_unscaled
        )

    # CALCULATE CLASSIFICATION METRICS

    def evaluate_model(self,model,X_test,y_test):

        # Predictions

        y_pred = model.predict(
            X_test
        )

        # Probability predictions

        if hasattr(
            model,
            "predict_proba"
        ):

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = model.decision_function(
                X_test
            )

            y_prob = (
                1 /
                (
                    1 +
                    __import__("numpy").exp(
                        -y_score
                    )
                )
            )

        # Confusion matrix

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        # Eight metrics

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        loss = log_loss(
            y_test,
            y_prob
        )

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Log Loss": loss
        }

    # TRAIN MODELS

    def train_models(self):

        (
            train_scaled,
            test_scaled,
            train_unscaled,
            test_unscaled
        ) = self.load_data()

        # SCALED DATA

        X_train_scaled = train_scaled.drop(
            columns=["is_canceled"]
        )

        y_train_scaled = train_scaled[
            "is_canceled"
        ]

        X_test_scaled = test_scaled.drop(
            columns=["is_canceled"]
        )

        y_test_scaled = test_scaled[
            "is_canceled"
        ]

        # UNSCALED DATA

        X_train_unscaled = train_unscaled.drop(
            columns=["is_canceled"]
        )

        y_train_unscaled = train_unscaled[
            "is_canceled"
        ]

        X_test_unscaled = test_unscaled.drop(
            columns=["is_canceled"]
        )

        y_test_unscaled = test_unscaled[
            "is_canceled"
        ]

       
        # FINAL MODEL
    
        # Use the exact finalized parameters from model-experiment notebooks from all models
        # KNN parameters are confirmed from our experiment:
        # n_neighbors = 11
        # weights = distance

        models = {

            "Logistic Regression": (
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),
                "scaled"
            ),

            "Decision Tree": (
                DecisionTreeClassifier(
                    random_state=42
                ),
                "unscaled"
            ),

            "Random Forest": (
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                ),
                "unscaled"
            ),

            "SVM": (
                SVC(
                    kernel="linear",
                    C=0.1,
                    probability=True,
                    random_state=42
                ),
                "scaled"
            ),

            "KNN": (
                KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance"
                ),
                "scaled"
            ),

            "Naive Bayes": (
                GaussianNB(),
                "scaled"
            )
        }

        results = []

        trained_models = {}

    
        # TRAIN EACH MODEL

        for model_name, (model,data_type) in models.items():

            print(f"Training {model_name}...")

            if data_type == "scaled":

                X_train = X_train_scaled
                y_train = y_train_scaled

                X_test = X_test_scaled
                y_test = y_test_scaled

            else:

                X_train = X_train_unscaled
                y_train = y_train_unscaled

                X_test = X_test_unscaled
                y_test = y_test_unscaled

            # Train

            model.fit(X_train,y_train)

            # Evaluate

            metrics = self.evaluate_model(model,X_test,y_test)

            metrics["Model"] = model_name

            results.append(metrics)

            trained_models[model_name] = model

        # MODEL COMPARISON
        
        results_df = pd.DataFrame(results)

        results_df = results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
                "Log Loss"
            ]
        ]

       
        # SELECT BEST MODEL
    
        # F1 is used as the primary selection metric because this is a cancellation classification 
        # problem and we care about both precision and recall.

        best_model_name = (
            results_df
            .sort_values(
                by="F1 Score",
                ascending=False
            )
            .iloc[0]["Model"]
        )

        best_model = trained_models[
            best_model_name
        ]

        print("\nModel Comparison:")

        print(results_df.round(4))

        print("\nBest Model:")

        print(best_model_name)

       
        # SAVE MODEL REPORT

        results_df.to_csv(
            self.config.model_report_path,
            index=False
        )


        # SAVE BEST MODEL

        joblib.dump(
            best_model,
            self.config.trained_model_path
        )

        print("\nModel Trainer Completed")

        print("Model Report Saved At:",
            self.config.model_report_path
        )

        print("Best Model Saved At:",
            self.config.trained_model_path
        )

        return (
            results_df,
            best_model_name
        )